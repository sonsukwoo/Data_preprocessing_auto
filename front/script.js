const $ = (id) => document.getElementById(id);
const API_BASE = window.location.protocol === "file:" ? "http://localhost:8000" : "/api";

const form = $("run-form");
const toast = $("toast");
const healthDot = $("health-dot");
const healthText = $("health-text");

const scriptEl = $("script");
const messagesEl = $("messages");
const fileInput = $("file-input");
const uploadStatus = $("upload-status");
const fileSummary = $("file-summary");
const dropzone = $("dropzone");
const outputFormatSelect = $("output_format");
const llmModelSelect = $("llm_model");
const coderModelSelect = $("coder_model");
const refactorCountInlineEl = $("refactor-count-inline");
const refactorDetailsEl = $("refactor-details");
const refactorEventsEl = $("refactor-events");
const progressEl = $("progress");
const stageTextEl = $("stage-text");
const stageDetailEl = $("stage-detail");
const stepperEl = $("stepper");
const downloadsEl = $("downloads");
const downloadsNoteEl = $("downloads-note");
// TEMP: tool calls debug panel (remove later)
const toolCallsEl = $("tool-calls");
const toolCallsNoteEl = $("tool-calls-note");

let selectedFiles = [];
let currentStage = "queued";
let hasRefactored = false;
let refactorEvents = [];

function renderToolCalls(toolCalls) {
  if (toolCallsEl) toolCallsEl.innerHTML = "";
  if (toolCallsNoteEl) toolCallsNoteEl.textContent = "";
  const calls = Array.isArray(toolCalls) ? toolCalls : [];
  if (!toolCallsEl) return;
  if (calls.length === 0) {
    if (toolCallsNoteEl) toolCallsNoteEl.textContent = "선택된 툴 없음";
    return;
  }
  if (toolCallsNoteEl) toolCallsNoteEl.textContent = `총 ${calls.length}개`;
  calls.forEach((tool) => {
    const li = document.createElement("li");
    const name = tool?.name || "tool";
    const reasonText = typeof tool?.reason === "string" ? tool.reason.trim() : "";
    li.textContent = reasonText ? `${name} - ${reasonText}` : name;
    toolCallsEl.appendChild(li);
  });
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("bad status");
    healthDot.classList.remove("offline");
    healthDot.classList.add("online");
    healthText.textContent = "백엔드 연결됨";
  } catch {
    healthDot.classList.remove("online");
    healthDot.classList.add("offline");
    healthText.textContent = "백엔드 연결 안 됨";
  }
}

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 2500);
}

function _formatCount(n) {
  try {
    return new Intl.NumberFormat().format(n);
  } catch {
    return String(n);
  }
}

function updateFileSummary(files) {
  const list = Array.from(files || []);
  if (!fileSummary) return;
  if (list.length === 0) {
    fileSummary.textContent = "선택된 파일 없음";
    return;
  }

  const first = list[0];
  const rel = first.webkitRelativePath || "";
  const root = rel ? rel.split("/")[0] : "";

  if (list.length === 1) {
    fileSummary.textContent = first.name;
    return;
  }
  if (root) {
    fileSummary.textContent = `${root} (${_formatCount(list.length)}개)`;
  } else {
    fileSummary.textContent = `${_formatCount(list.length)}개 파일`;
  }
}

function getSelectedFiles() {
  const fromInput = Array.from(fileInput?.files || []);
  return fromInput.length > 0 ? fromInput : selectedFiles;
}

function setRefactorCount(n) {
  const val = String(n ?? 0);
  if (refactorCountInlineEl) refactorCountInlineEl.textContent = val;
  if (Number(n || 0) > 0) hasRefactored = true;
  if (refactorDetailsEl) {
    refactorDetailsEl.classList.toggle("has-refactors", Number(n || 0) > 0);
  }
  // stage UI may need to update visuals based on count
  setStage(currentStage, stageDetailEl?.textContent || "");
}

const MAIN_STAGES = [
  "queued",
  "inspecting",
  "sampling",
  "context",
  "analyzing",
  "tooling",
  "generating",
  "executing",
  "validating",
  "finalizing",
  "done",
];
const BRANCH_STAGE = "refactoring";

const STAGE_LABELS = {
  queued: "대기 중",
  inspecting: "입력 검사 중",
  sampling: "데이터 샘플링 중",
  context: "컨텍스트 구성 중",
  analyzing: "요구사항 정리 중",
  tooling: "툴 조사/전수 스캔 중",
  generating: "스크립트 생성 중",
  executing: "스크립트 실행 중",
  validating: "요구사항 검증 중",
  refactoring: "리팩트 중",
  finalizing: "결과 정리 중",
  done: "완료",
  error: "실패",
};

function setStage(stage, detail = "") {
  if (!progressEl || !stepperEl) return;

  const prevStage = currentStage;
  currentStage = stage;
  if (stage === BRANCH_STAGE) hasRefactored = true;
  progressEl.classList.toggle("failed", stage === "error");
  if (refactorDetailsEl) {
    refactorDetailsEl.classList.toggle("active-refactor", stage === BRANCH_STAGE);
  }

  // main flow: queued → analyzing → sampling → generating → executing → finalizing → done
  const mainIndexByStage = new Map(MAIN_STAGES.map((s, i) => [s, i]));
  const mainIndex = mainIndexByStage.has(stage)
    ? mainIndexByStage.get(stage)
    : // if branching refactor (or unknown), keep last main stage or fall back to validating/executing
      mainIndexByStage.get(mainIndexByStage.has(prevStage) ? prevStage : "validating") ??
      mainIndexByStage.get("executing");

  const steps = Array.from(stepperEl.querySelectorAll(".step"));
  steps.forEach((el) => {
    const s = el.getAttribute("data-stage") || "";
    const isMain = mainIndexByStage.has(s);

    el.classList.remove("active", "done");
    if (stage === "error") return;

    if (isMain) {
      const idx = mainIndexByStage.get(s);
      if (typeof idx === "number" && typeof mainIndex === "number") {
        // done for all earlier steps
        if (idx < mainIndex) el.classList.add("done");
        // active only when currently on that main stage (not during branch)
        if (stage === s) el.classList.add("active");
        if (stage === "done" && s === "done") el.classList.add("active");
        if (stage === "done") el.classList.add("done");
      }
    } else if (s === BRANCH_STAGE) {
      if (stage === BRANCH_STAGE) {
        el.classList.add("active");
      } else if (hasRefactored) {
        el.classList.add("done");
      }
    }
  });

  const branchArrows = Array.from(stepperEl.querySelectorAll(".branch-arrow"));
  branchArrows.forEach((arrow) => {
    arrow.classList.remove("active", "done");
    if (stage === BRANCH_STAGE) {
      arrow.classList.add("active");
    } else if (hasRefactored) {
      arrow.classList.add("done");
    }
  });

  if (stageTextEl) stageTextEl.textContent = STAGE_LABELS[stage] || stage;
  if (stageDetailEl) stageDetailEl.textContent = detail || "";

  if (stage === BRANCH_STAGE && prevStage !== BRANCH_STAGE) {
    const clean = String(detail || "").trim();
    const ts = new Date();
    const timeLabel = ts.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const entry = clean ? `[${timeLabel}] ${clean}` : `[${timeLabel}] 리팩트 시작`;
    if (refactorEvents.length === 0 || refactorEvents[refactorEvents.length - 1] !== entry) {
      refactorEvents = [...refactorEvents, entry].slice(-20);
      if (refactorEventsEl) {
        refactorEventsEl.innerHTML = "";
        refactorEvents.forEach((t) => {
          const li = document.createElement("li");
          li.textContent = t;
          refactorEventsEl.appendChild(li);
        });
      }
    }
  }
}

function renderResult(data) {
  const imports = data.imports || "";
  const code = data.code || "";
  if (scriptEl) scriptEl.textContent = [imports.trim(), code.trim()].filter(Boolean).join("\n\n");

  renderToolCalls(data.tool_calls);

  if (downloadsEl) downloadsEl.innerHTML = "";
  if (downloadsNoteEl) downloadsNoteEl.textContent = "";
  const runId = data.run_id || "";
  const files = Array.isArray(data.output_files) ? data.output_files : [];
  if (downloadsEl && runId && files.length > 0) {
    files.forEach((name) => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = `${API_BASE}/downloads/${encodeURIComponent(runId)}/${encodeURIComponent(name)}`;
      a.textContent = name;
      a.setAttribute("download", name);
      li.appendChild(a);

      const isMarkdown = name.toLowerCase().endsWith(".md");
      const isInternalTrace = name.toLowerCase().includes("internal_trace");
      if (isMarkdown || isInternalTrace) {
        downloadsEl.appendChild(li);
        return;
      }

      const previewWrap = document.createElement("div");
      previewWrap.className = "preview";
      previewWrap.textContent = "미리보기 로딩 중...";
      li.appendChild(previewWrap);
      downloadsEl.appendChild(li);

      const previewUrl = `${API_BASE}/downloads/${encodeURIComponent(runId)}/${encodeURIComponent(name)}/preview?n=20`;
      fetch(previewUrl)
        .then((r) => {
          if (!r.ok) return r.text().then((t) => { throw new Error(t || r.statusText); });
          return r.json();
        })
        .then((payload) => {
          const rows = Array.isArray(payload?.rows) ? payload.rows : [];
          const cols = Array.isArray(payload?.columns) ? payload.columns : [];
          if (cols.length === 0) {
            previewWrap.textContent = "미리보기 없음";
            return;
          }
          const table = document.createElement("table");
          table.className = "preview-table";
          const thead = document.createElement("thead");
          const trh = document.createElement("tr");
          cols.forEach((c) => {
            const th = document.createElement("th");
            th.textContent = c;
            trh.appendChild(th);
          });
          thead.appendChild(trh);
          table.appendChild(thead);

          const tbody = document.createElement("tbody");
          rows.slice(0, 20).forEach((row) => {
            const tr = document.createElement("tr");
            cols.forEach((c) => {
              const td = document.createElement("td");
              const v = row && Object.prototype.hasOwnProperty.call(row, c) ? row[c] : "";
              td.textContent = v === null || v === undefined ? "" : String(v);
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
          });
          table.appendChild(tbody);

          previewWrap.innerHTML = "";
          previewWrap.appendChild(table);
        })
        .catch(() => {
          previewWrap.textContent = "미리보기 불러오기 실패";
        });
    });
  }

  messagesEl.innerHTML = "";
  (data.messages || []).forEach((m) => {
    const li = document.createElement("li");
    li.textContent = `${m.role}: ${m.content}`;
    messagesEl.appendChild(li);
  });
}

async function runWithStream(payload) {
  const res = await fetch(`${API_BASE}/run_stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  if (!res.body) throw new Error("Streaming not supported");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalData = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const msg = JSON.parse(trimmed);
      if (msg.type === "progress") {
        setRefactorCount(msg.iterations);
      } else if (msg.type === "stage") {
        setStage(msg.stage, msg.detail || "");
      } else if (msg.type === "tool_calls") {
        renderToolCalls(msg.tool_calls);
      } else if (msg.type === "final") {
        finalData = msg.data;
        setStage("done", "");
      } else if (msg.type === "error") {
        setStage("error", msg.detail || "");
        throw new Error(msg.detail || "Unknown error");
      }
    }
  }

  if (buffer.trim()) {
    const msg = JSON.parse(buffer.trim());
    if (msg.type === "final") finalData = msg.data;
    if (msg.type === "progress") setRefactorCount(msg.iterations);
    if (msg.type === "stage") setStage(msg.stage, msg.detail || "");
    if (msg.type === "tool_calls") renderToolCalls(msg.tool_calls);
    if (msg.type === "error") throw new Error(msg.detail || "Unknown error");
  }

  if (!finalData) throw new Error("No final response received");
  return finalData;
}

async function createS3UploadSession() {
  const res = await fetch(`${API_BASE}/s3/create_upload_session`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function presignS3Put(upload_id, files) {
  const body = {
    upload_id,
    files: files.map((f) => ({
      path: f.path,
      content_type: f.content_type || null,
    })),
  };
  const res = await fetch(`${API_BASE}/s3/presign_put`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function uploadToBackend(files) {
  const form = new FormData();
  for (const file of files) {
    const name = file.webkitRelativePath || file.name;
    form.append("files", file, name);
  }
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return await res.json(); // { path: "..." }
}

async function putToPresignedUrl(url, file, contentType) {
  const headers = {};
  if (contentType) headers["Content-Type"] = contentType;
  let res;
  try {
    res = await fetch(url, { method: "PUT", headers, body: file });
  } catch (err) {
    const msg = err?.message || String(err);
    throw new Error(
      `S3 업로드 실패(브라우저 차단): ${msg}. S3 버킷 CORS 설정이 필요할 수 있습니다. (자동으로 서버 업로드로 전환 가능)`
    );
  }
  if (!res.ok) throw new Error(`S3 업로드 실패: ${res.status}`);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question) {
    showToast("요청 문장을 입력하세요.");
    return;
  }

  const btn = $("run-btn");
  btn.disabled = true;
  btn.textContent = "실행 중...";

  try {
    setRefactorCount(0);
    refactorEvents = [];
    if (refactorEventsEl) refactorEventsEl.innerHTML = "";
    if (refactorDetailsEl) refactorDetailsEl.open = false;
    renderToolCalls([]);
    setStage("queued", "요청 준비 중");
    let finalQuestion = question;

    // 파일이 있으면 먼저 업로드 → 경로를 질문에 포함
    const files = getSelectedFiles();
    if (files.length > 0) {
      try {
        uploadStatus.textContent = "S3 업로드 준비 중...";

        const session = await createS3UploadSession();
        const uploadFiles = files.map((file) => ({
          file,
          path: file.webkitRelativePath || file.name,
          content_type: file.type || "application/octet-stream",
        }));

        const presigned = await presignS3Put(session.upload_id, uploadFiles);
        uploadStatus.textContent = `S3 업로드 중... (0/${uploadFiles.length})`;

        const urlByPath = new Map(presigned.items.map((it) => [it.path, it.url]));
        let done = 0;
        // Upload sequentially to keep it simple/reliable.
        for (const f of uploadFiles) {
          const url = urlByPath.get(f.path);
          if (!url) throw new Error(`presign url not found: ${f.path}`);
          await putToPresignedUrl(url, f.file, f.content_type);
          done += 1;
          uploadStatus.textContent = `S3 업로드 중... (${done}/${uploadFiles.length})`;
        }

        uploadStatus.textContent = "S3 업로드 완료";
        const s3Uri = `s3://${session.bucket}/${session.prefix}`;
        finalQuestion = `${s3Uri} ${question}`;
      } catch (err) {
        console.warn("S3 upload failed; falling back to backend upload:", err);
        uploadStatus.textContent = "S3 업로드 실패. 서버 업로드로 전환 중...";
        const uploaded = await uploadToBackend(files);
        uploadStatus.textContent = "서버 업로드 완료";
        finalQuestion = `${uploaded.path} ${question}`;
      }
    }

    const payload = {
      question: finalQuestion,
      max_iterations: Number($("max_iterations").value) || 3,
      output_format: outputFormatSelect.value,
    };
    const llmModel = llmModelSelect?.value?.trim();
    if (llmModel) payload.llm_model = llmModel;
    const coderModel = coderModelSelect?.value?.trim();
    if (coderModel) payload.coder_model = coderModel;

    let data;
    try {
      data = await runWithStream(payload);
    } catch (err) {
      // Fallback to non-streaming endpoint.
      setStage("finalizing", "");
      const res = await fetch(`${API_BASE}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      data = await res.json();
      setStage("done", "");
    }
    renderResult(data);
    showToast("완료!");
  } catch (err) {
    console.error(err);
    setStage("error", err.message || "");
    showToast(`실패: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "실행";
  }
});

// 드래그 앤 드롭 업로드 UI
["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const dt = e.dataTransfer;
  if (!dt?.files?.length) return;
  selectedFiles = Array.from(dt.files);
  updateFileSummary(selectedFiles);
  try {
    const next = new DataTransfer();
    selectedFiles.forEach((f) => next.items.add(f));
    fileInput.files = next.files;
  } catch {
    // 일부 브라우저에서는 files 직접 설정이 제한될 수 있음. selectedFiles를 사용해 계속 진행.
  }
});

fileInput.addEventListener("change", () => {
  selectedFiles = Array.from(fileInput.files || []);
  updateFileSummary(selectedFiles);
});

// 초기 헬스체크
checkHealth();
setStage("queued", "대기 중");
setRefactorCount(0);
updateFileSummary([]);
