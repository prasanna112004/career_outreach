/**
 * Best-effort scrape of LinkedIn job view. DOM classes change; adjust as needed.
 */
function scrapeJob() {
  const isLoginWall =
    location.pathname.includes("/login") ||
    !!document.querySelector('input[name="session_key"], input[name="session_password"]') ||
    /sign in/i.test(document.body?.innerText || "");

  if (isLoginWall) {
    return {
      title: "",
      company: "",
      description: "",
      url: location.href,
      login_wall: true,
      source: "login_wall",
    };
  }

  const expandBtn = document.querySelector(
    'button[aria-label*="Show more"], button[aria-label*="See more"], .jobs-description__footer-button'
  );
  if (expandBtn && typeof expandBtn.click === "function") {
    try {
      expandBtn.click();
    } catch (_e) {
      // best effort
    }
  }

  const title =
    document.querySelector("h1")?.innerText?.trim() ||
    document.querySelector('[data-test-job-title]')?.innerText?.trim() ||
    document.querySelector(".job-details-jobs-unified-top-card__job-title")?.innerText?.trim() ||
    document.querySelector(".jobs-unified-top-card__job-title")?.innerText?.trim() ||
    "";

  let company = "";
  const companyEl = document.querySelector(
    ".jobs-unified-top-card__company-name a, " +
      ".job-details-jobs-unified-top-card__company-name a, " +
      "a.job-details-jobs-unified-top-card__company-name, " +
      ".job-details-jobs-unified-top-card__company-name"
  );
  if (companyEl) company = companyEl.innerText.trim();

  const descEl = document.querySelector(
    ".jobs-description-content__text, " +
      ".jobs-box__html-content, " +
      "article.jobs-description__container, " +
      ".jobs-description, " +
      "#job-details, " +
      ".jobs-search__job-details--wrapper, " +
      ".scaffold-layout__detail"
  );
  let description = descEl ? descEl.innerText.trim() : "";
  if (!description) {
    const metaDesc =
      document.querySelector('meta[name="description"]')?.getAttribute("content") ||
      document.querySelector('meta[property="og:description"]')?.getAttribute("content") ||
      "";
    description = metaDesc.trim();
  }

  if (!description || description.length < 120) {
    const bodyText = (document.body?.innerText || "").replace(/\s+/g, " ").trim();
    const idx = bodyText.toLowerCase().search(
      /about the job|job description|responsibilities|requirements/
    );
    if (idx >= 0) {
      description = bodyText.slice(idx, idx + 3500).trim();
    } else {
      description = bodyText.slice(0, 3500).trim();
    }
  }

  // LinkedIn search result pages often carry active job id as query param.
  const params = new URLSearchParams(location.search);
  const currentJobId = params.get("currentJobId");
  const canonicalUrl = currentJobId
    ? `https://www.linkedin.com/jobs/view/${currentJobId}/`
    : location.href;

  return {
    title,
    company,
    description,
    url: canonicalUrl,
    login_wall: false,
    source: location.pathname.includes("/jobs/search-results") ? "search_results" : "job_view",
  };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "SCRAPE_JOB") {
    try {
      sendResponse({ ok: true, data: scrapeJob() });
    } catch (e) {
      sendResponse({ ok: false, error: String(e) });
    }
  }
  return true;
});
