import { Circle, CircleMarker, Polyline, Popup } from 'react-leaflet'

// Blue -> yellow -> red as probability rises. Kept perceptually ordered so the
// densest part of the origin cloud is unmistakable at a glance.
function heatColor(v) {
  const stops = [
    [0.00, [ 60, 110, 200]],
    [0.35, [ 90, 190, 190]],
    [0.65, [230, 200,  90]],
    [1.00, [215,  60,  50]],
  ]
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [a, ca] = stops[i - 1]
      const [b, cb] = stops[i]
      const f = (v - a) / (b - a || 1)
      const c = ca.map((x, k) => Math.round(x + f * (cb[k] - x)))
      return `rgb(${c[0]},${c[1]},${c[2]})`
    }
  }
  return 'rgb(215,60,50)'
}

/** Gridded origin probability. Cells below `min` are skipped so the map stays readable. */
export function OriginHeatmap({ drift, min = 0.18 }) {
  if (!drift?.heatmap) return null
  const { grid, bounds } = drift.heatmap
  const [latMin, latMax, lonMin, lonMax] = bounds
  const nLat = grid.length
  const nLon = grid[0]?.length || 0
  if (!nLat || !nLon) return null

  const dLat = (latMax - latMin) / nLat
  const dLon = (lonMax - lonMin) / nLon
  const cells = []
  for (let i = 0; i < nLat; i++) {
    for (let j = 0; j < nLon; j++) {
      const v = grid[i][j]
      if (v < min) continue
      cells.push(
        <Circle
          key={`${i}-${j}`}
          center={[latMin + (i + 0.5) * dLat, lonMin + (j + 0.5) * dLon]}
          radius={Math.max(dLat * 111320 * 0.52, 50)}
          pathOptions={{ color: heatColor(v), weight: 0,
                         fillColor: heatColor(v),
                         // Kept translucent: these cells overlap heavily, and a
                         // higher alpha stacks into an opaque wall that hides the
                         // SAR scene and the vessel tracks underneath it.
                         fillOpacity: 0.06 + 0.30 * v }}
        />
      )
    }
  }
  return <>{cells}</>
}

/** The back-tracked discharge path — a line, because a moving ship lays a line. */
export function OriginTrack({ drift }) {
  const pts = drift?.origin_track || []
  if (pts.length < 2) return null
  const line = pts.map((p) => [p[0], p[1]])
  return (
    <>
      <Polyline positions={line}
                pathOptions={{ color: '#fff', weight: 4, opacity: 0.55,
                               dashArray: '6,7' }} />
      <CircleMarker center={line[0]} radius={6}
                    pathOptions={{ color: '#fff', weight: 2,
                                   fillColor: '#d7391c', fillOpacity: 0.95 }}>
        <Popup>
          <b>Estimated discharge path</b><br />
          Earliest end of the window.<br />
          <span style={{ color: '#8b98a5' }}>
            Spread {drift.spread_km} km RMS over {drift.n_particles} members
          </span>
        </Popup>
      </CircleMarker>
    </>
  )
}

/** One forecast frame of drifting particles, driven by the timeline slider. */
export function DriftParticles({ drift, frameIdx }) {
  const frames = drift?.forecast?.frames || []
  if (!frames.length) return null
  const frame = frames[Math.min(frameIdx, frames.length - 1)]
  // Thinned: 600 individual markers is more than Leaflet redraws smoothly while
  // a user scrubs the timeline.
  const pts = frame.particles.filter((_, i) => i % 3 === 0)
  return (
    <>
      {pts.map((p, i) => (
        <CircleMarker key={i} center={p} radius={2}
                      pathOptions={{ color: '#2f81f7', weight: 0,
                                     fillColor: '#2f81f7', fillOpacity: 0.5 }} />
      ))}
    </>
  )
}

// Rank colours shared with the suspect table.
export const SUSPECT_COLORS = ['#d7191c', '#fd8d3c', '#fecc5c', '#74add1', '#a6a6a6']
export const suspectColor = (rank) =>
  SUSPECT_COLORS[Math.min(rank - 1, SUSPECT_COLORS.length - 1)]

/** AIS tracks, coloured by suspicion rank. */
export function AisTracks({ attribution, selected, onSelect, maxShow = 8 }) {
  const suspects = (attribution?.suspects || []).slice(0, maxShow)
  return (
    <>
      {suspects.map((s) => {
        const pts = (s.track || []).map((p) => [p[0], p[1]])
        if (pts.length < 2) return null
        const isSel = selected?.mmsi === s.mmsi
        const col = suspectColor(s.rank)
        return (
          <Polyline
            key={s.mmsi}
            positions={pts}
            pathOptions={{ color: col, weight: isSel ? 5 : 2.5,
                           opacity: isSel ? 1 : 0.75 }}
            eventHandlers={{ click: () => onSelect(isSel ? null : s) }}
          >
            <Popup>
              <b>#{s.rank} · MMSI {s.mmsi}</b><br />
              {s.name || '(unnamed)'}<br />
              <span style={{ color: col, fontWeight: 700 }}>
                {s.attribution_pct}%
              </span> attribution<br />
              <span style={{ color: '#8b98a5' }}>
                closest {s.min_dist_km} km · track match {s.track_match}
              </span>
            </Popup>
          </Polyline>
        )
      })}
    </>
  )
}
