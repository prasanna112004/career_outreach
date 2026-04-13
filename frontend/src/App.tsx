import { useMemo, useState } from "react";
import {
  analyzeMatch,
  generateContent,
  refineContent,
  type OutputType,
  type Tone,
} from "./api";

export default function App() {
  const [jobUrl, setJobUrl] = useState("");
  const [resume, setResume] = useState("");
  const [jdPaste, setJdPaste] = useState("");
  const [tone, setTone] = useState<Tone>("professional");
  const [outputType, setOutputType] = useState<OutputType>("email");
  const [chatModel, setChatModel] = useState("llama-3.1-8b-instant");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [match, setMatch] = useState<{
    match_score: number;
    missing_skills: string[];
    strong_skills: string[];
  } | null>(null);

  const [generated, setGenerated] = useState("");
  const [contextPreview, setContextPreview] = useState("");

  const canMatch = useMemo(
    () => resume.trim().length >= 20 && jdPaste.trim().length >= 20,
    [resume, jdPaste]
  );

  async function onAnalyzeMatch() {
    setErr(null);
    setLoading(true);
    try {
      const m = await analyzeMatch(resume, jdPaste, chatModel);
      setMatch(m);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onGenerate() {
    setErr(null);
    setLoading(true);
    try {
      const res = await generateContent({
        job_url: jobUrl.trim() || undefined,
        resume_text: resume.trim() || undefined,
        use_resume: !!resume.trim(),
        tone,
        output_type: outputType,
        chat_model: chatModel,
      });
      setGenerated(res.generated_text);
      setContextPreview(res.retrieved_context_preview);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h1 style={{ marginTop: 0 }}>Career Outreach Assistant</h1>
      <p style={{ color: "#475569", fontSize: "0.95rem" }}>
        Dev server proxies <code>/api/*</code> → FastAPI on <code>:8000</code>. Run{" "}
        <code>uvicorn api.main:app --reload</code> from <code>cold_email_ai/</code>.
      </p>

      <div className="row">
        <label>Job posting URL</label>
        <input
          style={{ width: "100%", padding: "0.5rem" }}
          value={jobUrl}
          onChange={(e) => setJobUrl(e.target.value)}
          placeholder="https://..."
        />
      </div>

      <div className="row">
        <label>Resume text</label>
        <textarea rows={8} value={resume} onChange={(e) => setResume(e.target.value)} />
      </div>

      <div className="row">
        <label>Job description (paste for match analysis — optional if URL works)</label>
        <textarea rows={6} value={jdPaste} onChange={(e) => setJdPaste(e.target.value)} />
      </div>

      <div className="row" style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
        <div>
          <label>Tone</label>
          <select value={tone} onChange={(e) => setTone(e.target.value as Tone)}>
            <option value="professional">professional</option>
            <option value="friendly">friendly</option>
            <option value="direct">direct</option>
          </select>
        </div>
        <div>
          <label>Output</label>
          <select
            value={outputType}
            onChange={(e) => setOutputType(e.target.value as OutputType)}
          >
            <option value="email">Email</option>
            <option value="linkedin">LinkedIn DM</option>
            <option value="cover_letter">Cover letter</option>
          </select>
        </div>
        <div>
          <label>Groq model</label>
          <input value={chatModel} onChange={(e) => setChatModel(e.target.value)} />
        </div>
      </div>

      {err && (
        <div className="row" style={{ color: "#b91c1c", fontSize: "0.9rem" }}>
          {err}
        </div>
      )}

      <div className="row" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" disabled={loading || !canMatch} onClick={() => void onAnalyzeMatch()}>
          Analyze match
        </button>
        <button type="button" disabled={loading || !jobUrl.trim()} onClick={() => void onGenerate()}>
          Generate
        </button>
      </div>

      {match && (
        <div className="row">
          <h3>Match</h3>
          <div className="progress">
            <div style={{ width: `${match.match_score}%` }} />
          </div>
          <p>
            <strong>{match.match_score}%</strong>
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
            <div>
              <div style={{ fontWeight: 600, marginBottom: "0.35rem" }}>Strong skills</div>
              <div className="pills">
                {match.strong_skills.map((s) => (
                  <span className="pill" key={s}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <div style={{ fontWeight: 600, marginBottom: "0.35rem" }}>Missing vs JD</div>
              <div className="pills">
                {match.missing_skills.map((s) => (
                  <span className="pill warn" key={s}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {generated && (
        <div className="row">
          <h3>Output</h3>
          <textarea rows={14} value={generated} onChange={(e) => setGenerated(e.target.value)} />
          <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.5rem" }}>
            <button
              type="button"
              className="secondary"
              disabled={loading}
              onClick={async () => {
                setLoading(true);
                try {
                  const r = await refineContent(generated, "Make shorter.", chatModel);
                  setGenerated(r.refined_text);
                } catch (e) {
                  setErr(String(e));
                } finally {
                  setLoading(false);
                }
              }}
            >
              Make shorter
            </button>
            <button
              type="button"
              className="secondary"
              disabled={loading}
              onClick={async () => {
                setLoading(true);
                try {
                  const r = await refineContent(
                    generated,
                    "Improve tone to be more professional.",
                    chatModel
                  );
                  setGenerated(r.refined_text);
                } catch (e) {
                  setErr(String(e));
                } finally {
                  setLoading(false);
                }
              }}
            >
              Improve tone
            </button>
          </div>
        </div>
      )}

      {contextPreview && (
        <div className="row">
          <h3>Retrieved context</h3>
          <textarea rows={8} readOnly value={contextPreview} />
        </div>
      )}
    </div>
  );
}
