import { useEffect, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const EXAMPLES = [
  "Un film d'une élégance rare, chaque scène est précise et émouvante.",
  "La mise en scène est confuse et les dialogues sonnent faux du début à la fin.",
  "J'étais sceptique, mais le rythme et les acteurs m'ont complètement embarqué."
];

function getApiUrl(path) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

function clampConfidence(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Number((numericValue * 100).toFixed(1))));
}

function normalizeLabelKey(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

function getProbability(probabilities, expectedLabel) {
  const normalizedExpectedLabel = normalizeLabelKey(expectedLabel);

  for (const [label, probability] of Object.entries(probabilities || {})) {
    if (normalizeLabelKey(label) === normalizedExpectedLabel) {
      return probability;
    }
  }

  return 0;
}

function formatPercent(value) {
  return `${clampConfidence(value)} %`;
}

function formatDuration(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "Non disponible";
  }
  return `${numericValue.toFixed(1)} ms`;
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
            ? "Le modèle est prêt à analyser votre avis."
            : "Le service est joignable, le modèle termine son chargement."
        });
      } catch (fetchError) {
        if (!isMounted) {
          return;
        }
        setHealth({
          loading: false,
          ready: false,
          message: "Le service est indisponible pour le moment."
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
        throw new Error(data.detail || "Analyse impossible");
      }

      setResult(data);
    } catch (requestError) {
      setResult(null);
      setError(requestError.message || "Analyse impossible");
    } finally {
      setLoading(false);
    }
  }

  const positive = normalizeLabelKey(result?.label) === "positif";
  const confidence = result ? clampConfidence(result.confidence) : 0;

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <main className="layout">
        <section className="hero-panel">
          <div className="eyebrow">AvisSense</div>
          <h1>
            Analysez un avis de film
            <span> en quelques secondes.</span>
          </h1>
          <p className="hero-copy">
            Écrivez ou collez un avis en français, lancez l'analyse, puis consultez
            un résultat simple : plutôt positif ou plutôt négatif, avec un niveau de confiance.
          </p>

          <div className="status-strip">
            <div className={`status-pill ${health.ready ? "ready" : "pending"}`}>
              {health.loading ? "Vérification du service..." : health.message}
            </div>
            <span>Choisissez un exemple ou saisissez votre propre avis.</span>
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
            <label htmlFor="review-input">Votre avis</label>
            <textarea
              id="review-input"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Écrivez ici votre avis de film en français..."
              rows={8}
            />

            <div className="analysis-actions">
              <button type="submit" className="primary-button" disabled={loading}>
                {loading ? "Analyse en cours..." : "Analyser l'avis"}
              </button>
              <span>{text.trim().length} caractères</span>
            </div>
          </form>

          <section className="result-card">
            <div className="result-header">
              <span className="result-title">Verdict</span>
              {result ? (
                <span className={`result-badge ${positive ? "positive" : "negative"}`}>
                  {positive ? "Positif" : "Négatif"}
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
                    <span>Niveau de confiance</span>
                    <strong>{formatPercent(result.confidence)}</strong>
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
                    <strong>{formatPercent(getProbability(result.probabilities, "positif"))}</strong>
                  </article>
                  <article>
                    <span>Négatif</span>
                    <strong>{formatPercent(getProbability(result.probabilities, "negatif"))}</strong>
                  </article>
                  <article>
                    <span>Temps de réponse</span>
                    <strong>{formatDuration(result.processing_time_ms)}</strong>
                  </article>
                </div>
              </>
            ) : (
              <p className="placeholder-copy">
                Collez un avis, cliquez sur le bouton d'analyse, puis consultez le verdict
                et le niveau de confiance affichés à droite.
              </p>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}
