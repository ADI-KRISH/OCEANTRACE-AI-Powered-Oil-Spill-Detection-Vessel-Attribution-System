import { suspectColor } from './DriftLayers'

// The headline output of the whole system: who could have done it, and why.
// Every row carries its evidence inline — the ranking is never shown as a bare
// score, because a number without a reason is not something anyone can act on.

export default function SuspectPanel({ attribution, selected, onSelect }) {
  if (!attribution) {
    return (
      <div className="empty">
        Run the full pipeline to rank suspect vessels.
      </div>
    )
  }
  const suspects = attribution.suspects || []
  if (!suspects.length) {
    return <div className="empty">No candidate vessels near the estimated origin.</div>
  }

  return (
    <>
      <div className="mm" style={{ marginBottom: 8 }}>
        {attribution.n_candidates} candidates from {attribution.n_vessels_screened} vessels
        {' · '}{attribution.scorer}
      </div>

      {suspects.map((s) => {
        const col = suspectColor(s.rank)
        const isSel = selected?.mmsi === s.mmsi
        return (
          <div key={s.mmsi}
               className={`slick ${isSel ? 'sel' : ''}`}
               style={isSel ? { borderColor: col, boxShadow: `0 0 0 1px ${col} inset` } : null}
               onClick={() => onSelect(isSel ? null : s)}>
            <div className="hd">
              <span className="id" style={{ color: col }}>#{s.rank}</span>
              <span className="area" style={{ color: col }}>
                {s.attribution_pct}%
              </span>
            </div>
            <div className="nm">{s.name || '(unnamed)'}</div>
            <div className="mm">
              MMSI {s.mmsi}
              {s.length_m ? ` · ${s.length_m} m` : ''}
              {' · '}closest {s.min_dist_km} km
            </div>

            <ul>{s.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>

            {s.learned_rank != null && s.learned_rank !== s.rank && (
              <div className="uncertain-note">
                Learned re-ranker places this vessel at #{s.learned_rank}.
                The transparent score above is the primary answer.
              </div>
            )}
          </div>
        )
      })}

      <div className="uncertain-note" style={{ marginTop: 10 }}>
        Attribution is correlation, not proof. This ranks which vessels
        <em> could </em> have produced the slick and states why — it does not
        establish that any of them did.
      </div>
    </>
  )
}
