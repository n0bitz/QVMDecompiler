const defaultInput = `\
int f(unsigned int x) {
    if(x < 2) {
        return x;
    }
    return f(x - 1) + f(x - 2);
}
`;

const config = {
  settings: {
    showPopoutIcon: false,
  },
  content: [
    {
      type: "row",
      content: [
        {
          type: "component",
          componentName: "inputComponent",
          title: "Input",
          isClosable: false,
          componentState: {
            input: defaultInput,
          },
        },
        {
          type: "stack",
          content: [
            {
              type: "component",
              componentName: "graphComponent",
              title: "Graph",
              isClosable: false,
            },
            {
              type: "component",
              componentName: "disassemblyComponent",
              title: "Disassembly",
              isClosable: false,
            },
            {
              type: "component",
              componentName: "assemblyComponent",
              title: "Assembly",
              isClosable: false,
            },
          ],
        },
      ],
    },
  ],
};

let savedState = localStorage.getItem("savedState");
let layout;
if (savedState !== null) {
  layout = new GoldenLayout(JSON.parse(savedState));
} else {
  layout = new GoldenLayout(config);
}

const createEditor = (container, mode, readOnly) => {
  container.on("resize", () => {
    window.dispatchEvent(new Event("resize"));
  });

  let editor = ace.edit(container.getElement()[0], {
    showPrintMargin: false,
    readOnly: readOnly,
    highlightActiveLine: !readOnly,
    highlightGutterLine: !readOnly,
    theme: "ace/theme/nord_dark",
    mode: mode,
  });

  if (readOnly) {
    editor.renderer.$cursorLayer.element.style.display = "none";
  }

  return editor;
};

layout.registerComponent(
  "inputComponent",
  function (container, componentState) {
    let editor = createEditor(container, "ace/mode/c_cpp", false);
    editor.session.setValue(componentState.input);

    let timeout;
    editor.session.on("change", () => {
      clearTimeout(timeout);
      timeout = setTimeout(update, 1000);
    });

    let lastStartTime = -1;
    async function checkRestart() {
      const response = await fetch("/status");
      let { startTime } = JSON.parse(await response.text());
      if (startTime != lastStartTime) {
        lastStartTime = startTime;
        await update();
      }
    }
    checkRestart();
    setInterval(checkRestart, 2000);

    async function update() {
      let input = editor.getValue();
      container.extendState({
        input: input,
      });

      layout.eventHub.emit("disassemblyUpdate", "Compiling...");
      layout.eventHub.emit("assemblyUpdate", "Compiling...");

      const response = await fetch("/compile", {
        method: "POST",
        body: input,
      });
      let { disassembly, graph, assembly } = JSON.parse(await response.text());

      layout.eventHub.emit("disassemblyUpdate", disassembly);
      layout.eventHub.emit("graphUpdate", graph);
      layout.eventHub.emit("assemblyUpdate", assembly);
    }
  }
);

layout.registerComponent(
  "disassemblyComponent",
  function (container, componentState) {
    let editor = createEditor(container, "ace/mode/c_cpp", true);
    editor.renderer.setShowGutter(false);

    layout.eventHub.on("disassemblyUpdate", (disassembly) => {
      editor.session.setValue(disassembly);
    });
  }
);

layout.registerComponent(
  "graphComponent",
  function (container, componentState) {
    let element = container.getElement();
    element.addClass("graph");

    layout.eventHub.on("graphUpdate", (graph) => {
      let svg = document.adoptNode(
        new DOMParser().parseFromString(graph, "image/svg+xml").documentElement
      );

      element.empty();
      element.append(svg);

      const setup = () => {
        svgPanZoom(svg, {
          minZoom: 0.1,
          zoomScaleSensitivity: 0.3,
        });
      };

      if (container.isHidden) {
        container.on("shown", setup);
      } else {
        setup();
      }
    });
  }
);

layout.registerComponent(
  "assemblyComponent",
  function (container, componentState) {
    let editor = createEditor(container, "ace/mode/assembly_x86", true);
    layout.eventHub.on("assemblyUpdate", (assembly) => {
      editor.session.setValue(assembly);
    });
  }
);

layout.on("stateChanged", () => {
  localStorage.setItem("savedState", JSON.stringify(layout.toConfig()));
});

layout.init();
