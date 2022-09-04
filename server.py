import json
import tempfile
import quatch
from pathlib import Path
from time import time
from bottle import Bottle, request, static_file
from qvm_decompiler.decompile import decompile
from qvm_decompiler.qvm import Qvm
from qvm_decompiler.cfg import build_cfg
from qvm_decompiler.graph import graph_function


class Server:
    def __init__(self):
        self.app = Bottle()
        self.app.route("/", "GET", self.index)
        self.app.route("/compile", "POST", self.compile)
        self.app.route("/<path:path>", "GET", self.static)

    def run(self):
        self.app.run(host="localhost", port=8080, reloader=True)

    def index(self):
        return static_file("index.html", "static")

    def static(self, path):
        return static_file(path, "static")

    def compile(self):
        source = request.body.read().decode()

        quatch_qvm = quatch.Qvm()
        try:
            output = quatch_qvm.add_c_code(source)
        except quatch._compile.CompilerError as e:
            output = str(e)
            return json.dumps({k: output for k in ["disassembly", "assembly", "decompilation", "output"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir).joinpath("input.c")
            asm_path = Path(tmpdir).joinpath("output.asm")
            qvm_path = Path(tmpdir).joinpath("output.qvm")

            input_path.write_text(source)
            quatch._compile.compile_c_file(input_path, asm_path)
            assembly = asm_path.read_text()

            quatch_qvm.write(qvm_path)
            qvm = Qvm(qvm_path)

        disassembly = "\n".join(str(ins) for ins in qvm.instructions)

        cfg = build_cfg(qvm)
        graph = graph_function(cfg[0]).to_str().decode()
        decompilation = decompile(cfg)

        return json.dumps(
            {
                "disassembly": disassembly,
                "graph": graph,
                "assembly": assembly,
                "decompilation": decompilation,
                "output": output,
            }
        )


if __name__ == "__main__":
    Server().run()
