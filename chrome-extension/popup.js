function $(id) {
  return document.getElementById(id);
}

function setStatus(t) {
  $("status").textContent = t || "";
}

function setResumeFileStatus(t) {
  $("resumeFileStatus").textContent = t || "";
}

async function parseResumeFile(apiBase, file) {
  const form = new FormData();
  form.append("resume_file", file, file.name);
  const r = await fetch(`${apiBase}/extract-resume`, {
    method: "POST",
    body: form,
  });
  const txt = await r.text();
  if (!r.ok) throw new Error(txt || `HTTP ${r.status}`);
  const data = JSON.parse(txt);
  return (data.resume_text || "").trim();
}

$("loadPage").addEventListener("click", async () => {
  setStatus("Reading tab…");
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    setStatus("No active tab.");
    return;
  }
  const tryScrape = async () => chrome.tabs.sendMessage(tab.id, { type: "SCRAPE_JOB" });

  try {
    let res;
    try {
      res = await tryScrape();
    } catch (_firstErr) {
      // Fallback: inject the content script on-demand and retry once.
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"],
      });
      await new Promise((r) => setTimeout(r, 200));
      res = await tryScrape();
    }

    if (!res?.ok) {
      setStatus(res?.error || "Could not scrape. Open a LinkedIn job page.");
      return;
    }
    const d = res.data;
    if (d?.login_wall) {
      setStatus("LinkedIn sign-in page detected. Sign in and open a specific job post.");
      return;
    }
    const desc = (d?.description || "").trim();
    window.__scrapedJob = d;
    if (desc.length < 80) {
      setStatus("Loaded page, but JD is short. Paste JD manually for best results.");
    } else {
      setStatus(`Loaded: ${d.title || "?"} @ ${d.company || "?"}`);
    }
  } catch (e) {
    setStatus("Scrape failed — open a LinkedIn job page, then click Load again.");
  }
});

$("resumeFile").addEventListener("change", async () => {
  const apiBase = $("apiBase").value.replace(/\/$/, "");
  const f = $("resumeFile").files?.[0];
  if (!f) {
    setResumeFileStatus("");
    return;
  }
  setResumeFileStatus(`Parsing ${f.name}...`);
  try {
    const parsed = await parseResumeFile(apiBase, f);
    if (!parsed || parsed.length < 20) {
      setResumeFileStatus("Could not extract enough text. You can paste resume manually.");
      return;
    }
    $("resume").value = parsed;
    setResumeFileStatus(`Loaded ${f.name} (${parsed.length} chars extracted).`);
  } catch (e) {
    setResumeFileStatus("Resume parse failed. Ensure FastAPI is running.");
    setStatus(String(e));
  }
});

$("generate").addEventListener("click", async () => {
  const apiBase = $("apiBase").value.replace(/\/$/, "");
  let resume = $("resume").value.trim();
  const jdManual = $("jdManual").value.trim();
  const tone = $("tone").value;
  const outputType = $("outputType").value;
  const chatModel = $("model").value.trim() || "llama-3.1-8b-instant";

  const job = window.__scrapedJob;
  const effectiveDescription = jdManual || (job?.description || "").trim();
  if (!effectiveDescription || effectiveDescription.length < 40) {
    setStatus("Load from LinkedIn or paste job description manually.");
    return;
  }
  if (resume.length < 20) {
    const file = $("resumeFile").files?.[0];
    if (file) {
      try {
        setStatus("Parsing resume file…");
        resume = await parseResumeFile(apiBase, file);
        $("resume").value = resume;
      } catch (e) {
        setStatus("Resume parse failed. Paste resume text or check API.");
        $("out").textContent = String(e);
        return;
      }
    }
    if (resume.length < 20) {
      setStatus("Upload resume file or paste resume text.");
      return;
    }
  }

  setStatus("Generating…");
  $("out").textContent = "";

  const body = {
    job_title: (job?.title || "").trim() || "LinkedIn Role",
    company: (job?.company || "").trim() || "Company",
    job_description_text: effectiveDescription,
    resume_text: resume,
    use_resume: true,
    tone,
    output_type: outputType,
    chat_model: chatModel,
  };

  try {
    const r = await fetch(`${apiBase}/generate-content`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    if (!r.ok) {
      setStatus(`Error ${r.status}`);
      $("out").textContent = text;
      return;
    }
    const data = JSON.parse(text);
    setStatus("Done.");
    $("out").textContent = data.generated_text || "";
  } catch (e) {
    setStatus("Network error — is FastAPI running?");
    $("out").textContent = String(e);
  }
});
