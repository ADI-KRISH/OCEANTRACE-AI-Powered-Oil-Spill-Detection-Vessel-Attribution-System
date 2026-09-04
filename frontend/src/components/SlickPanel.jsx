// Per-slick characterisation. Every number that feeds a later stage is shown,
// and the age carries its interval and confidence inline -- CLAUDE.md requires
// that age is never presented as exact, so the UI must not render it as one.

function ageText(s) {
  if (s.age_estimate_h == null) return 'n/a'
  if (s.age_saturated === 'upper') return `> ${s.age_estimate_h} h`
  if (s.age_saturated === 'lower') return `< ${s.age_estimate_h} h`
  return `~${s.age_estimate_h} h (${s.age_range_h[0]}–${s.age_range_h[1]})`
}

export default function SlickPanel({ scene, selected, onSelect }) {
  if (!scene) {
    return <div className="empty">Run a detection to see results.</div>
  }
  if (!scene.slicks.length) {
    return <div className="empty">No oil slicks detected in this scene.</div>
  }

  return (
    <>
      <h2>Detected slicks ({scene.n_slicks})</h2>
      {scene.slicks.map((s) => (
        <div key={s.id}
             className={`slick ${selected?.id === s.id ? 'sel' : ''}`}
             onClick={() => onSelect(selected?.id === s.id ? null : s)}>
          <div className="hd">
            <span className="id">Slick #{s.id}</span>
            <span className="area">{s.area_km2} km²</span>
          </div>
          <div style={{ marginTop: 5 }}>
            <span className={`badge ${s.oil_likelihood}`}>
              {s.oil_likelihood.replace('_', ' ')}
            </span>
          </div>

          <div className="grid">
            <span className="k">Long axis</span>
            <span className="v">{s.orientation_deg}° (undirected)</span>
            <span className="k">Aspect</span>
            <span className="v">{s.elongation}:1</span>
            <span className="k">Contrast</span>
            <span className="v">{s.contrast_db} dB</span>
            <span className="k">Solidity</span>
            <span className="v">{s.solidity}</span>
            <span className="k">Extent</span>
            <span className="v">{s.major_axis_m} × {s.minor_axis_m} m</span>
            <span className="k">Confidence</span>
            <span className="v">{s.confidence >= 0 ? s.confidence : 'n/a'}</span>
            <span className="k">Age</span>
            <span className="v">{ageText(s)}</span>
          </div>

          <div className="uncertain-note">
            Age is an estimate, not a measurement ({s.age_confidence} confidence).
            Since t ~ r⁴, a small error in extent becomes a large error in age.
          </div>

          {s.notes?.length > 0 && (
            <ul>{s.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
          )}
        </div>
      ))}
    </>
  )
}
