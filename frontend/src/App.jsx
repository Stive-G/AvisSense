import { useEffect, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const EXAMPLES = [
  "Un film d'une elegance rare, chaque scene est precise et emouvante.",
  "La mise en scene est confuse et les dialogues sonnent faux du debut a la fin.",
  "J'etais sceptique, mais le rythme et les acteurs m'ont completement embarque."
];

function getApiUrl(path) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

function clampConfidence(value) {
  return Math.max(0, Math.min(100, Number((value * 100).toFixed(1))));
}

export default function App() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState({ loading: true, ready: false, message: "" });

  useEffect(() => {
    let isMounted = true;

    async function fetchHealth() {
      try {
        const response = await fetch(getApiUrl("/health"));
        if (!response.ok) {
          throw new Error("API indisponible");
        }
        const data = await response.json();
        if (!isMounted) {
          return;
        }
        setHealth({
          loading: false,
          ready: Boolean(data.model_loaded),
          message: data.model_loaded
            ? "Modele pret pour l'inference"
            : "API joignable, modele encore en chargement"
        });
      } catch (fetchError) {
        if (!isMounted) {
          return;
        }
        setHealth({
          loading: false,
          ready: false,
          message: "Impossible de joindre l'API"
        });
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
      setError("Saisis un avis avant de lancer l'analyse.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(getApiUrl("/predict"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: trimmed })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction impossible");
      }

      setResult(data);
    } catch (requestError) {
      setResult(null);
      setError(requestError.message || "Prediction impossible");
    } finally {
      setLoading(false);
    }
  }

  const positive = result?.label === "positif";
  const confidence = result ? clampConfidence(result.confidence) : 0;

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <main className="layout">
        <section className="hero-panel">
          <div className="eyebrow">AvisSense / Sentiment cinema francais</div>
          <h1>
            Une interface front qui vend le modele
            <span> sans cacher l'incertitude.</span>
          </h1>
          <p className="hero-copy">
            Ce front React parle a une API FastAPI hebergee sur Hugging Face et
            renvoie un verdict lisible, un niveau de confiance et un etat de service.
          </p>

          <div className="status-strip">
            <div className={`status-pill ${health.ready ? "ready" : "pending"}`}>
              {health.loading ? "Verification de l'API..." : health.message}
            </div>
            <a href={getApiUrl("/docs")} target="_blank" rel="noreferrer">
              Voir la doc API
            </a>
          </div>

          <div className="example-grid">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                className="example-card"
                onClick={() => setText(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </section>

        <section className="workbench">
          <form className="analysis-card" onSubmit={handleSubmit}>
            <div className="card-topline">
              <span>POST /predict</span>
              <span>JSON</span>
            </div>

            <label htmlFor="review-input">Avis a analyser</label>
            <textarea
              id="review-input"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Ecris un avis de film en francais..."
              rows={8}
            />

            <div className="analysis-actions">
              <button type="submit" className="primary-button" disabled={loading}>
                {loading ? "Analyse en cours..." : "Lancer l'analyse"}
              </button>
              <span>{text.trim().length} caracteres</span>
            </div>
          </form>

          <section className="result-card">
            <div className="result-header">
              <span className="result-title">Verdict</span>
              {result ? (
                <span className={`result-badge ${positive ? "positive" : "negative"}`}>
                  {positive ? "Positif" : "Negatif"}
                </span>
              ) : (
                <span className="result-badge idle">En attente</span>
              )}
            </div>

            {error ? <p className="error-box">{error}</p> : null}

            {result ? (
              <>
                <div className="meter-block">
                  <div className="meter-labels">
                    <span>Confiance du modele</span>
                    <strong>{confidence}%</strong>
                  </div>
                  <div className="meter-track">
                    <div
                      className={`meter-fill ${positive ? "positive" : "negative"}`}
                      style={{ width: `${confidence}%` }}
                    />
                  </div>
                </div>

                <div className="probability-grid">
                  <article>
                    <span>Positif</span>
                    <strong>{clampConfidence(result.probabilities.positif)}%</strong>
                  </article>
                  <article>
                    <span>Negatif</span>
                    <strong>{clampConfidence(result.probabilities.negatif)}%</strong>
                  </article>
                  <article>
                    <span>Temps de reponse</span>
                    <strong>{result.processing_time_ms} ms</strong>
                  </article>
                </div>
              </>
            ) : (
              <p className="placeholder-copy">
                Colle un avis, lance l'analyse, puis utilise la confiance pour juger si le
                verdict est net ou ambigu.
              </p>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}
