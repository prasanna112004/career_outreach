const BASE = "/api";

export type Tone = "professional" | "friendly" | "direct";
export type OutputType = "email" | "linkedin" | "cover_letter";

export async function analyzeMatch(resumeText: string, jdText: string, chatModel: string) {
  const r = await fetch(`${BASE}/analyze-match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      resume_text: resumeText,
      job_description_text: jdText,
      chat_model: chatModel,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{
    match_score: number;
    missing_skills: string[];
    strong_skills: string[];
  }>;
}

export async function generateContent(payload: {
  job_url?: string;
  job_title?: string;
  company?: string;
  job_description_text?: string;
  resume_text?: string;
  use_resume: boolean;
  tone: Tone;
  output_type: OutputType;
  chat_model: string;
}) {
  const r = await fetch(`${BASE}/generate-content`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{
    generated_text: string;
    quality_score: number;
    retrieved_context_preview: string;
  }>;
}

export async function refineContent(
  original: string,
  instruction: string,
  chatModel: string
) {
  const r = await fetch(`${BASE}/refine-content`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      original_generated_text: original,
      instruction,
      chat_model: chatModel,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ refined_text: string }>;
}
