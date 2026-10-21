document.addEventListener("DOMContentLoaded", function () {
  // grab elements
  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("fileInput");
  var docList = document.getElementById("docList");
  var indexedChunksCount = document.getElementById("indexedChunksCount");

  var btnIndex = document.getElementById("btnIndex");
  var chunkSizeInput = document.getElementById("chunkSize");
  var chunkOverlapInput = document.getElementById("chunkOverlap");
  var btnClearDb = document.getElementById("btnClearDb");

  var queryForm = document.getElementById("queryForm");
  var queryInput = document.getElementById("queryInput");
  var btnSubmit = document.getElementById("btnSubmit");
  var topKSelect = document.getElementById("topKSelect");

  var welcomeBox = document.getElementById("welcomeBox");
  var answerCard = document.getElementById("answerCard");
  var userQueryDisplay = document.getElementById("userQueryDisplay");
  var answerBody = document.getElementById("answerBody");
  var responseMeta = document.getElementById("responseMeta");

  var sourcesDrawer = document.getElementById("sourcesDrawer");
  var sourcesToggle = document.getElementById("sourcesToggle");
  var sourcesContent = document.getElementById("sourcesContent");
  var sourceCount = document.getElementById("sourceCount");
  var toast = document.getElementById("toast");

  // toast notification
  function showToast(msg, type) {
    toast.textContent = msg;
    toast.className = "toast show " + (type || "");
    setTimeout(function () {
      toast.className = "toast";
    }, 3000);
  }

  // escape html
  function escapeHtml(text) {
    if (!text) return "";
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // load status from server
  function loadStatus() {
    fetch("/api/status")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        indexedChunksCount.textContent = data.indexed_chunks || 0;
      })
      .catch(function () {});
  }

  // load uploaded documents list
  function loadDocuments() {
    fetch("/api/documents")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        var docs = data.documents || [];
        if (docs.length === 0) {
          docList.innerHTML = '<p class="muted">No documents yet.</p>';
          return;
        }

        var html = "";
        for (var i = 0; i < docs.length; i++) {
          var d = docs[i];
          html += '<div class="file-item">';
          html += '  <span class="name">' + escapeHtml(d.name) + '</span>';
          html += '  <span class="meta">' + d.size_kb + ' KB</span>';
          html += '  <button class="delete-btn" data-name="' + escapeHtml(d.name) + '">&times;</button>';
          html += '</div>';
        }
        docList.innerHTML = html;

        // attach delete handlers
        var deleteBtns = docList.querySelectorAll(".delete-btn");
        for (var j = 0; j < deleteBtns.length; j++) {
          deleteBtns[j].addEventListener("click", function () {
            var fname = this.getAttribute("data-name");
            if (confirm('Delete "' + fname + '"?')) {
              deleteDocument(fname);
            }
          });
        }
      })
      .catch(function () {
        docList.innerHTML = '<p class="muted">Failed to load.</p>';
      });
  }

  // upload a file
  function uploadFile(file) {
    var formData = new FormData();
    formData.append("file", file);

    return fetch("/api/upload", { method: "POST", body: formData })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data.success) {
          showToast("Uploaded " + file.name, "success");
        } else {
          showToast("Upload failed", "error");
        }
        loadDocuments();
        loadStatus();
      })
      .catch(function (err) {
        showToast("Upload error: " + err.message, "error");
      });
  }

  // delete a document
  function deleteDocument(fname) {
    var formData = new FormData();
    formData.append("filename", fname);

    fetch("/api/delete_document", { method: "POST", body: formData })
      .then(function () {
        showToast("Deleted " + fname);
        loadDocuments();
        loadStatus();
      })
      .catch(function () {
        showToast("Delete failed", "error");
      });
  }

  // dropzone click
  dropzone.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    var files = fileInput.files;
    for (var i = 0; i < files.length; i++) {
      uploadFile(files[i]);
    }
    fileInput.value = "";
  });

  // drag and drop
  dropzone.addEventListener("dragover", function (e) {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", function () {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    var files = e.dataTransfer.files;
    for (var i = 0; i < files.length; i++) {
      uploadFile(files[i]);
    }
  });

  // index button
  btnIndex.addEventListener("click", function () {
    var chunkSize = parseInt(chunkSizeInput.value) || 1000;
    var chunkOverlap = parseInt(chunkOverlapInput.value) || 200;

    btnIndex.disabled = true;
    btnIndex.textContent = "Indexing...";

    fetch("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap })
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data.success) {
          showToast("Indexed " + data.chunks_indexed + " chunks in " + data.time_seconds + "s", "success");
        } else {
          showToast(data.message || "Indexing failed", "error");
        }
        loadStatus();
      })
      .catch(function (err) {
        showToast("Indexing failed: " + err.message, "error");
      })
      .finally(function () {
        btnIndex.disabled = false;
        btnIndex.textContent = "Index Documents";
      });
  });

  // clear db
  btnClearDb.addEventListener("click", function () {
    if (!confirm("Clear the entire vector database?")) return;
    fetch("/api/clear", { method: "POST" })
      .then(function () {
        showToast("Database cleared");
        loadStatus();
      })
      .catch(function () {
        showToast("Failed to clear", "error");
      });
  });

  // query form
  queryForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var query = queryInput.value.trim();
    if (!query) return;

    var topK = parseInt(topKSelect.value) || 3;

    btnSubmit.disabled = true;
    btnSubmit.textContent = "Thinking...";

    fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, top_k: topK })
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data.detail) {
          showToast(data.detail, "error");
          return;
        }

        welcomeBox.style.display = "none";
        answerCard.style.display = "flex";

        userQueryDisplay.textContent = query;
        answerBody.textContent = data.answer;
        responseMeta.textContent = data.response_time + "s | " + data.model_used;

        // sources
        var sources = data.sources || [];
        sourceCount.textContent = sources.length;

        if (sources.length > 0) {
          sourcesDrawer.style.display = "block";
          sourcesContent.style.display = "none";

          var html = "";
          for (var i = 0; i < sources.length; i++) {
            var s = sources[i];
            var name = (s.metadata && s.metadata.source_name) || "Document";
            var page = (s.metadata && s.metadata.page) || 1;
            var score = s.similarity ? (s.similarity * 100).toFixed(1) + "%" : "";
            html += '<div class="source-item">';
            html += '  <div class="source-header"><strong>' + escapeHtml(name) + ' (p.' + page + ')</strong> <span>' + score + '</span></div>';
            html += '  <div class="source-snippet">' + escapeHtml(s.text) + '</div>';
            html += '</div>';
          }
          sourcesContent.innerHTML = html;
        } else {
          sourcesDrawer.style.display = "none";
        }
      })
      .catch(function (err) {
        showToast("Error: " + err.message, "error");
      })
      .finally(function () {
        btnSubmit.disabled = false;
        btnSubmit.textContent = "Ask";
      });
  });

  // toggle sources
  sourcesToggle.addEventListener("click", function () {
    var visible = sourcesContent.style.display !== "none";
    sourcesContent.style.display = visible ? "none" : "flex";
    sourcesToggle.textContent = (visible ? "Show" : "Hide") + " sources (" + sourceCount.textContent + ")";
  });

  // init
  loadStatus();
  loadDocuments();
});
