import { useEffect, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const MAX_CHARACTERS = 5000;

const EXAMPLES = [
  {
    tone: "positif",
    label: "Une belle surprise",
    text: "Un film d'une élégance rare, chaque scène est précise et émouvante."
  },
  {
    tone: "negatif",
    label: "Une vraie déception",
    text: "La mise en scène est confuse et les dialogues sonnent faux du début à la fin."
  },
  {
    tone: "positif",
    label: "Un avis nuancé",
    text: "J'étais sceptique, mais le rythme et les acteurs m'ont complètement embarqué."
  }
];

function Icon({ name }) {
  const paths = {
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    check: <path d="m5 12 4 4L19 6" />,
    film: (
      <>
        <rect x="3" y="5" width="18" height="14" rx="2" />
        <path d="M7 5v14M17 5v14M3 9h4m10 0h4M3 15h4m10 0h4" />
      </>
    ),
    spark: <path d="m12 3 1.4 4.2L18 9l-4.6 1.8L12 15l-1.4-4.2L6 9l4.6-1.8L12 3Zm6 11 .7 2.3L21 17l-2.3.7L18 20l-.7-2.3L15 17l2.3-.7L18 14Z" />,
    trash: (
      <>
        <path d="M4 7h16m-10 4v5m4-5v5M9 7l1-3h4l1 3m3 0-1 13H7L6 7" />
      </>
    )
  };

  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      {paths[name]}
    </svg>
  );
}

function getApiUrl(path) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

function normalizeLabelKey(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

function clampConfidence(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue)
    ? Math.max(0, Math.min(100, Number((numericValue * 100).toFixed(1))))
    : 0;
}

function getProbability(probabilities, expectedLabel) {
  const normalizedExpectedLabel = normalizeLabelKey(expectedLabel);
  const entry = Object.entries(probabilities || {}).find(
    ([label]) => normalizeLabelKey(label) === normalizedExpectedLabel
  );
  return entry ? entry[1] : 0;
}

function formatPercent(value) {
  return `${clampConfidence(value)} %`;
}

function formatDuration(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? `${numericValue.toFixed(1)} ms` : "—";
}

export default function App() {
  const [text, setText] = useState(EXAMPLES[0].text);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState({ loading: true, ready: false });

  useEffect(() => {
    let isMounted = true;

    async function fetchHealth() {
      try {
        const response = await fetch(getApiUrl("/health"));
        if (!response.ok) throw new Error("API indisponible");
        const data = await response.json();
        if (isMounted) setHealth({ loading: false, ready: Boolean(data.model_loaded) });
      } catch {
        if (isMounted) setHealth({ loading: false, ready: false });
      }
    }

    fetchHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = text.trim();

    if (!trimmed) {
      setError("Saisissez un avis avant de lancer l'analyse.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(getApiUrl("/predict"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analyse impossible");
      setResult(data);
    } catch (requestError) {
      setResult(null);
      setError(requestError.message || "Analyse impossible");
    } finally {
      setLoading(false);
    }
  }

  function chooseExample(example) {
    setText(example.text);
    setError("");
  }

  const positive = normalizeLabelKey(result?.label) === "positif";
  const confidence = result ? clampConfidence(result.confidence) : 0;
  const characterCount = text.length;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="AvisSense, accueil">
          <span className="brand-mark"><Icon name="film" /></span>
          <span>AvisSense</span>
        </a>
        <div className={`service-status ${health.ready ? "online" : ""}`}>
          <span className="status-dot" />
          {health.loading ? "Connexion au modèle" : health.ready ? "Modèle opérationnel" : "Service indisponible"}
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow"><Icon name="spark" /> Analyse de sentiment en français</div>
            <h1>Comprenez ce que vos avis <em>veulent vraiment dire.</em></h1>
            <p>
              AvisSense détecte instantanément le sentiment d'un avis cinéma et vous
              livre un verdict clair, accompagné de son niveau de confiance.
            </p>
            <div className="hero-points">
              <span><Icon name="check" /> Résultat instantané</span>
              <span><Icon name="check" /> Analyse en français</span>
              <span><Icon name="check" /> Score de confiance</span>
            </div>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="quote-card quote-card-back">
              <span>« Une réalisation remarquable. »</span>
            </div>
            <div className="quote-card quote-card-front">
              <div className="quote-topline"><span>Verdict détecté</span><strong>Positif</strong></div>
              <div className="quote-mark">“</div>
              <p>Un film sensible, porté par des acteurs exceptionnels.</p>
              <div className="confidence-preview">
                <span><i /> Confiance</span>
                <strong>96,8 %</strong>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace">
          <div className="workspace-heading">
            <div>
              <span className="section-index">01 / Analyse</span>
              <h2>Testez un avis</h2>
            </div>
            <p>Collez votre texte ou partez d'un exemple.</p>
          </div>

          <div className="workspace-grid">
            <form className="editor-card" onSubmit={handleSubmit}>
              <div className="card-heading">
                <div>
                  <span className="card-kicker">Votre texte</span>
                  <h3>Que pensez-vous du film ?</h3>
                </div>
                <button
                  className="icon-button"
                  type="button"
                  onClick={() => {
                    setText("");
                    setResult(null);
                    setError("");
                  }}
                  aria-label="Effacer le texte"
                  title="Effacer"
                >
                  <Icon name="trash" />
                </button>
              </div>

              <textarea
                id="review-input"
                value={text}
                maxLength={MAX_CHARACTERS}
                onChange={(event) => setText(event.target.value)}
                placeholder="Écrivez ou collez un avis de film en français..."
                rows={8}
              />

              <div className="editor-footer">
                <span className={characterCount > MAX_CHARACTERS * 0.9 ? "count warning" : "count"}>
                  {characterCount.toLocaleString("fr-FR")} / {MAX_CHARACTERS.toLocaleString("fr-FR")}
                </span>
                <button type="submit" className="primary-button" disabled={loading}>
                  {loading ? <span className="spinner" /> : <Icon name="spark" />}
                  {loading ? "Analyse en cours" : "Analyser cet avis"}
                  {!loading && <Icon name="arrow" />}
                </button>
              </div>
            </form>

            <section className={`result-card ${result ? (positive ? "is-positive" : "is-negative") : ""}`}>
              <div className="card-heading">
                <div>
                  <span className="card-kicker">Résultat</span>
                  <h3>Verdict de l'analyse</h3>
                </div>
                <span className={`verdict-pill ${result ? (positive ? "positive" : "negative") : "idle"}`}>
                  <span />
                  {result ? (positive ? "Positif" : "Négatif") : "En attente"}
                </span>
              </div>

              {error ? (
                <div className="error-box">{error}</div>
              ) : result ? (
                <div className="result-content">
                  <div className="score">
                    <span className="score-label">Confiance du modèle</span>
                    <strong>{confidence.toLocaleString("fr-FR")}<small>%</small></strong>
                    <div className="meter-track">
                      <div className="meter-fill" style={{ width: `${confidence}%` }} />
                    </div>
                  </div>
                  <div className="metric-grid">
                    <article>
                      <span>Positif</span>
                      <strong>{formatPercent(getProbability(result.probabilities, "positif"))}</strong>
                    </article>
                    <article>
                      <span>Négatif</span>
                      <strong>{formatPercent(getProbability(result.probabilities, "negatif"))}</strong>
                    </article>
                    <article>
                      <span>Temps d'analyse</span>
                      <strong>{formatDuration(result.processing_time_ms)}</strong>
                    </article>
                  </div>
                </div>
              ) : (
                <div className="empty-result">
                  <div className="empty-icon"><Icon name="spark" /></div>
                  <h4>Votre résultat apparaîtra ici</h4>
                  <p>Lancez une analyse pour connaître le sentiment et le score de confiance.</p>
                </div>
              )}
            </section>
          </div>

          <div className="examples-section">
            <span className="examples-label">Besoin d'inspiration ?</span>
            <div className="example-grid">
              {EXAMPLES.map((example, index) => (
                <button
                  key={example.text}
                  type="button"
                  className={`example-card ${text === example.text ? "active" : ""}`}
                  onClick={() => chooseExample(example)}
                >
                  <span className={`example-number ${example.tone}`}>0{index + 1}</span>
                  <span>
                    <strong>{example.label}</strong>
                    <small>{example.text}</small>
                  </span>
                  <Icon name="arrow" />
                </button>
              ))}
            </div>
          </div>
        </section>
      </main>

      <footer>
        <a className="brand" href="/" aria-label="AvisSense, accueil">
          <span className="brand-mark"><Icon name="film" /></span>
          <span>AvisSense</span>
        </a>
        <p>Analyse de sentiment cinéma propulsée par DistilCamemBERT.</p>
        <span>Projet IA · 2026</span>
      </footer>
    </div>
  );
}
