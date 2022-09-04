let layout;

const componentRegistrations = [];
const registerComponent = (...args) => void componentRegistrations.push(args);

const defaultStackContent = [];
const registerStackComponent = (componentName, component) => {
  registerComponent(componentName, component);
  defaultStackContent.push({
    type: "component",
    componentName,
    title: `${componentName[0].toUpperCase()}${componentName.slice(1)}`,
    isClosable: false
  });
};

const defaultInput = `\
int f(unsigned int x) {
    if(x < 2) {
        return x;
    }
    return f(x - 1) + f(x - 2);
}
`;
const defaultConfig = {
  settings: {
    showPopoutIcon: false,
  },
  content: [
    {
      type: "row",
      content: [
        {
          type: "component",
          componentName: "input",
          title: "Input",
          isClosable: false,
          componentState: {
            input: defaultInput,
          },
        },
        {
          type: "stack",
          content: defaultStackContent,
        },
      ],
    },
  ],
};

const createEditor = (container, mode, readOnly) => {
  container.on("resize", () => {
    window.dispatchEvent(new Event("resize"));
  });

  const editor = ace.edit(container.getElement()[0], {
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

const getTextComponent = (language, sessionProps = {}) => function(container, componentState) {
    const editor = createEditor(container, `ace/mode/${language}`, true);
    for (const [k, v] of Object.entries(sessionProps)) editor.session[k] = v;
    layout.eventHub.on(`${componentState.componentName}Update`, (newValue) => {
      editor.session.setValue(newValue);
      if (container.isHidden) container.on("shown", () => void editor.resize());
    });
};

registerStackComponent("disassembly", getTextComponent("c_cpp", {gutterRenderer: {
  getWidth: (session, lastLineNumber, config) => 10 * config.characterWidth,
  getText: (session, row) => `0x${row.toString(16).padStart(8, "0")}`
}}));

for (const [name, language] of Object.entries({
  assembly: "assembly_x86",
  decompilation: "c_cpp",
  output: "text"
})) registerStackComponent(name, getTextComponent(language));

registerStackComponent("graph", function(container, componentState) {
    const element = container.getElement();
    element.addClass("graph");

    layout.eventHub.on("graphUpdate", (graph) => {
      element.empty();
      if (!graph) return;
      const svg = document.adoptNode(
        new DOMParser().parseFromString(graph, "image/svg+xml").documentElement
      );

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

registerComponent(
  "input",
  function (container, componentState) {
    const editor = createEditor(container, "ace/mode/c_cpp", false);
    editor.session.setValue(componentState.input);

    let timeout;
    editor.session.on("change", () => {
      clearTimeout(timeout);
      timeout = setTimeout(update, 1000);
    });

    async function update() {
      const input = editor.getValue();
      container.extendState({input});

      layout.eventHub.emit("disassemblyUpdate", "Compiling...");
      layout.eventHub.emit("assemblyUpdate", "Compiling...");
      layout.eventHub.emit("decompilationUpdate", "Compiling...");
      layout.eventHub.emit("outputUpdate", "Compiling...");
      layout.eventHub.emit("graphUpdate", "");

      const response = await fetch("/compile", {
        method: "POST",
        body: input,
      });
      for (const [k, v] of Object.entries(JSON.parse(await response.text())))
        layout.eventHub.emit(`${k}Update`, v);
    }

    update();
  }
);

(async function() {
  const savedState = JSON.parse(localStorage.getItem("savedState"));
  const layoutBuf = (new TextEncoder()).encode(JSON.stringify(defaultConfig));
  const layoutVersion =
    [...new Uint8Array(await crypto.subtle.digest("SHA-256", layoutBuf))]
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
  if (savedState && localStorage.getItem("layoutVersion") === layoutVersion) {
    layout = new GoldenLayout(savedState);
  } else {
    layout = new GoldenLayout(defaultConfig);
    localStorage.setItem("layoutVersion", layoutVersion);
  }
  for (const componentArgs of componentRegistrations) layout.registerComponent(...componentArgs);
  layout.on("stateChanged", () => {
    localStorage.setItem("savedState", JSON.stringify(layout.toConfig()));
  });
  layout.init();
})();
