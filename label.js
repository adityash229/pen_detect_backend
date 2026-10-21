(function () {
  "use strict";

  var GREEN = "#00c853";
  var MIN_BOX_SIZE = 4;

  function createElement(tag, attrs) {
    var element = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "text") {
        element.textContent = attrs[key];
      } else if (key === "className") {
        element.className = attrs[key];
      } else {
        element.setAttribute(key, attrs[key]);
      }
    });
    return element;
  }

  function setMessage(element, text, isError) {
    element.textContent = text;
    element.style.color = isError ? "#b00020" : "#0b6b2b";
  }

  function LabelCanvas(options) {
    this.root = options.root;
    this.endpoint = options.endpoint || "/api/label";
    this.boxes = [];
    this.imageFile = null;
    this.image = new Image();
    this.isDrawing = false;
    this.start = { x: 0, y: 0 };
    this.current = { x: 0, y: 0 };

    this.fileInput = createElement("input", {
      type: "file",
      accept: "image/*",
    });
    this.canvas = createElement("canvas", {
      "aria-label": "Pen label drawing canvas",
    });
    this.undoButton = createElement("button", { type: "button", text: "Undo" });
    this.submitButton = createElement("button", { type: "button", text: "Submit" });
    this.message = createElement("p", { role: "status" });

    this.ctx = this.canvas.getContext("2d");
    this.mount();
    this.bindEvents();
  }

  LabelCanvas.prototype.mount = function () {
    var toolbar = createElement("div", { className: "label-toolbar" });
    toolbar.appendChild(this.fileInput);
    toolbar.appendChild(this.undoButton);
    toolbar.appendChild(this.submitButton);

    this.canvas.style.display = "block";
    this.canvas.style.maxWidth = "100%";
    this.canvas.style.cursor = "crosshair";
    this.canvas.style.border = "1px solid #d0d7de";

    this.root.appendChild(toolbar);
    this.root.appendChild(this.canvas);
    this.root.appendChild(this.message);
  };

  LabelCanvas.prototype.bindEvents = function () {
    var self = this;

    this.fileInput.addEventListener("change", function (event) {
      self.loadFile(event.target.files && event.target.files[0]);
    });

    this.canvas.addEventListener("mousedown", function (event) {
      self.startDrawing(event);
    });
    this.canvas.addEventListener("mousemove", function (event) {
      self.moveDrawing(event);
    });
    window.addEventListener("mouseup", function (event) {
      self.finishDrawing(event);
    });

    this.undoButton.addEventListener("click", function () {
      self.undo();
    });
    this.submitButton.addEventListener("click", function () {
      self.submit();
    });
  };

  LabelCanvas.prototype.loadFile = function (file) {
    var self = this;
    if (!file) {
      return;
    }

    this.imageFile = file;
    this.boxes = [];
    this.image.onload = function () {
      self.canvas.width = self.image.naturalWidth;
      self.canvas.height = self.image.naturalHeight;
      self.redraw();
      setMessage(self.message, "Image loaded. Draw boxes around each pen.", false);
    };
    this.image.onerror = function () {
      setMessage(self.message, "Could not load image.", true);
    };
    this.image.src = URL.createObjectURL(file);
  };

  LabelCanvas.prototype.getCanvasPoint = function (event) {
    var rect = this.canvas.getBoundingClientRect();
    var scaleX = this.canvas.width / rect.width;
    var scaleY = this.canvas.height / rect.height;

    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  };

  LabelCanvas.prototype.startDrawing = function (event) {
    if (!this.imageFile) {
      setMessage(this.message, "Choose an image first.", true);
      return;
    }

    this.isDrawing = true;
    this.start = this.getCanvasPoint(event);
    this.current = this.start;
  };

  LabelCanvas.prototype.moveDrawing = function (event) {
    if (!this.isDrawing) {
      return;
    }

    this.current = this.getCanvasPoint(event);
    this.redraw();
    this.drawRect(this.makeBox(this.start, this.current));
  };

  LabelCanvas.prototype.finishDrawing = function (event) {
    if (!this.isDrawing) {
      return;
    }

    this.isDrawing = false;
    this.current = this.getCanvasPoint(event);

    var box = this.makeBox(this.start, this.current);
    if (box.w >= MIN_BOX_SIZE && box.h >= MIN_BOX_SIZE) {
      this.boxes.push(box);
      setMessage(this.message, this.boxes.length + " box(es) ready.", false);
    }
    this.redraw();
  };

  LabelCanvas.prototype.makeBox = function (start, end) {
    return {
      x: Math.round(Math.min(start.x, end.x)),
      y: Math.round(Math.min(start.y, end.y)),
      w: Math.round(Math.abs(end.x - start.x)),
      h: Math.round(Math.abs(end.y - start.y)),
    };
  };

  LabelCanvas.prototype.drawRect = function (box) {
    this.ctx.strokeStyle = GREEN;
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(box.x, box.y, box.w, box.h);
  };

  LabelCanvas.prototype.redraw = function () {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.imageFile) {
      this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
    }

    for (var i = 0; i < this.boxes.length; i += 1) {
      this.drawRect(this.boxes[i]);
    }
  };

  LabelCanvas.prototype.undo = function () {
    if (!this.boxes.length) {
      setMessage(this.message, "No boxes to undo.", true);
      return;
    }

    this.boxes.pop();
    this.redraw();
    setMessage(this.message, this.boxes.length + " box(es) ready.", false);
  };

  LabelCanvas.prototype.submit = async function () {
    if (!this.imageFile) {
      setMessage(this.message, "Choose an image before submitting.", true);
      return;
    }
    if (!this.boxes.length) {
      setMessage(this.message, "Draw at least one box before submitting.", true);
      return;
    }

    var formData = new FormData();
    formData.append("image", this.imageFile, this.imageFile.name || "label.jpg");
    formData.append("boxes", JSON.stringify(this.boxes));

    this.submitButton.disabled = true;
    setMessage(this.message, "Submitting labels...", false);

    try {
      var response = await fetch(this.endpoint, {
        method: "POST",
        body: formData,
      });
      var data = await response.json().catch(function () {
        return {};
      });

      if (!response.ok) {
        throw new Error(data.detail || "Label upload failed.");
      }

      setMessage(this.message, "Labels saved successfully.", false);
    } catch (error) {
      setMessage(this.message, error.message || "Label upload failed.", true);
    } finally {
      this.submitButton.disabled = false;
    }
  };

  function initLabelTool() {
    var root = document.getElementById("label-tool");
    if (!root) {
      root = createElement("div", { id: "label-tool" });
      document.body.appendChild(root);
    }

    return new LabelCanvas({ root: root, endpoint: "/api/label" });
  }

  window.LabelCanvas = LabelCanvas;
  window.initLabelTool = initLabelTool;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLabelTool);
  } else {
    initLabelTool();
  }
})();
