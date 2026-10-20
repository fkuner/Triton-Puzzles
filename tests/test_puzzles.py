import ast
import inspect
import json
import linecache
import os
import unittest
from pathlib import Path

import torch
from torch import Tensor
import triton
import triton.language as tl
from jaxtyping import Float32, Int32


def load_puzzles():
    path = Path(__file__).resolve().parents[1] / "Triton-Puzzles.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {
        "torch": torch,
        "Tensor": Tensor,
        "triton": triton,
        "tl": tl,
        "Float32": Float32,
        "Int32": Int32,
    }
    cases = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or "_kernel(" not in source:
            continue
        filename = f"{path}.cell_{index}.py"
        linecache.cache[filename] = (
            len(source), None, source.splitlines(True), filename
        )
        tree = ast.parse(source, filename=filename)
        definitions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.Assign))
        ]
        module = ast.Module(body=definitions, type_ignores=[])
        exec(compile(module, filename, "exec"), namespace)
        for node in tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not isinstance(call.func, ast.Name) or call.func.id != "test":
                continue
            keywords = {
                item.arg: ast.literal_eval(item.value) for item in call.keywords
            }
            sizes = keywords["nelem"]
            blocks = keywords.get("B", {"B0": 32}).copy()
            for axis in (1, 2):
                if f"N{axis}" in sizes:
                    blocks.setdefault(f"B{axis}", 32)
            cases.append((call.args[0].id, call.args[1].id, sizes, blocks))
    if len(cases) != 12:
        raise ValueError(f"Expected 12 puzzles, found {len(cases)}")
    return namespace, cases


class PuzzleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace, cls.cases = load_puzzles()
        default_device = (
            "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
        )
        cls.device = os.environ.get("PUZZLES_DEVICE", default_device)

    def setUp(self):
        torch.manual_seed(0)

    def check(self, name, inputs, expected, sizes, blocks):
        inputs = [value.to(self.device) for value in inputs]
        storage = torch.full((expected.numel() + 32,), 173.0, device=self.device)
        output = storage[: expected.numel()].view(expected.shape)
        grid = tuple(
            triton.cdiv(sizes.get(f"N{axis}", 1), blocks.get(f"B{axis}", 1))
            for axis in range(3)
        )
        self.namespace[name][grid](*inputs, output, **sizes, **blocks)
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-3, atol=1e-3)
        self.assertTrue(torch.all(storage[expected.numel() :] == 173.0).item())

    def test_notebook_specs(self):
        for name, spec_name, sizes, blocks in self.cases:
            with self.subTest(puzzle=name):
                spec = self.namespace[spec_name]
                inputs = []
                for parameter in inspect.signature(spec).parameters.values():
                    shape = [dim.size for dim in parameter.annotation.dims]
                    if parameter.annotation.dtypes[0] == "int32":
                        value = torch.randint(
                            -100000, 100000, shape, dtype=torch.int32
                        )
                    else:
                        value = torch.rand(shape) - 0.5
                    inputs.append(value)
                self.check(name, inputs, spec(*inputs), sizes, blocks)

    def test_vector_tails(self):
        for length in (1, 31, 33, 201):
            for name in ("add_kernel", "add_mask2_kernel"):
                with self.subTest(length=length, puzzle=name):
                    x = torch.randn(length)
                    self.check(
                        name, [x], x + 10, {"N0": length}, {"B0": 32}
                    )

    def test_outer_product_tails(self):
        x, y = torch.randn(37), torch.randn(19)
        for name in (
            "add_vec_kernel", "add_vec_block_kernel", "mul_relu_block_kernel"
        ):
            with self.subTest(puzzle=name):
                expected = x[None, :] + y[:, None]
                if name == "mul_relu_block_kernel":
                    expected = torch.relu(x[None, :] * y[:, None])
                self.check(
                    name, [x, y], expected,
                    {"N0": 37, "N1": 19}, {"B0": 16, "B1": 8},
                )

    def test_relu_backward_zero_and_tails(self):
        x, y, dz = torch.randn(19, 37), torch.randn(19), torch.randn(19, 37)
        x[:, 0] = 0
        y[0] = 0
        expected = dz * torch.where(x * y[:, None] > 0, y[:, None], 0.0)
        self.check(
            "mul_relu_block_back_kernel", [x, y, dz], expected,
            {"N0": 37, "N1": 19}, {"B0": 16, "B1": 8},
        )

    def test_row_reductions(self):
        for length in (1, 31, 33, 201):
            x = torch.randn(5, length) * 100
            for name in ("sum_kernel", "softmax_kernel"):
                with self.subTest(length=length, puzzle=name):
                    expected = x.sum(1) if name == "sum_kernel" else x.softmax(1)
                    self.check(
                        name, [x], expected,
                        {"N0": 5, "N1": 32, "T": length},
                        {"B0": 2, "B1": 32},
                    )

    def test_attention_tails_and_stability(self):
        for queries, keys in ((1, 1), (17, 33), (65, 101)):
            with self.subTest(queries=queries, keys=keys):
                q = torch.randn(queries) * 10
                k = torch.randn(keys) * 10
                v = torch.randn(keys)
                expected = (q[:, None] * k[None, :]).softmax(1) @ v
                self.check(
                    "flashatt_kernel", [q, k, v], expected,
                    {"N0": queries, "T": keys}, {"B0": 32},
                )

    def test_convolution_rectangular_images(self):
        x, k = torch.randn(3, 5, 7), torch.randn(3, 2)
        padded = torch.nn.functional.pad(x, (0, 1, 0, 2))
        expected = torch.nn.functional.conv2d(padded[:, None], k[None, None])[:, 0]
        self.check(
            "conv2d_kernel", [x, k], expected,
            {"N0": 3, "H": 5, "W": 7, "KH": 3, "KW": 2}, {"B0": 2},
        )

    def test_matmul_all_dimension_tails(self):
        x, y = torch.randn(3, 19, 35), torch.randn(3, 35, 23)
        self.check(
            "dot_kernel", [x, y], x @ y,
            {"N0": 19, "N1": 23, "N2": 3, "MID": 35},
            {"B0": 16, "B1": 16, "B2": 2, "B_MID": 16},
        )

    def test_quantized_matmul_full_bit_patterns(self):
        scale = torch.rand(19, 8)
        offset = torch.randint(-(2**31), 2**31, (19,), dtype=torch.int32)
        weight = torch.randint(-(2**31), 2**31, (19, 8), dtype=torch.int32)
        weight[0, 0] = -1
        weight[0, 1] = 0
        activation = torch.randn(64, 23)
        shifts = torch.arange(8) * 4
        unpacked_weight = ((weight[..., None] >> shifts) & 15).reshape(19, 64)
        unpacked_offset = ((offset[:, None] >> shifts) & 15).repeat_interleave(
            8, dim=1
        )
        restored_weight = scale.repeat_interleave(8, dim=1) * (
            unpacked_weight - unpacked_offset
        )
        expected = restored_weight @ activation
        for block_mid in (16, 64):
            with self.subTest(block_mid=block_mid):
                self.check(
                    "quant_dot_kernel", [scale, offset, weight, activation], expected,
                    {"N0": 19, "N1": 23, "MID": 64},
                    {"B0": 16, "B1": 16, "B_MID": block_mid},
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
