import { useEffect, useState } from 'react'
import { detect, getClasses, getModules, getRegions, runPipeline } from './api'
import SuspectPanel from './components/SuspectPanel'
import MapView from './components/MapView'
import ModuleStatus from './components/ModuleStatus'
import SlickPanel from './components/SlickPanel'

// Layers the spec calls for. `needs` names the module that would supply the data;
// a layer whose module is not built stays disabled and says why, rather than
// rendering an empty overlay that looks like "nothing detected".
const LAYERS = [
  { key: 'sar',        label: 'SAR scene',            needs: 'detection' },
  { key: 'mask',       label: 'Segmentation mask',    needs: 'detection' },
  { key: 'polygons',   label: 'Spill polygons',       needs: 'detection' },
  { key: 'truth',      label: 'Ground truth',         needs: 'detection', synthOnly: true },
  { key: 'originHeat', label: 'Origin heatmap',       needs: 'drift' },
  { key: 'driftAnim',  label: 'Drift particles',      needs: 'drift' },
  { key: 'aisTracks',  label: 'AIS tracks',           needs: 'attribution' },
]

export default function App() {
  const [modules, setModules] = useState(null)
  const [classes, setClasses] = useState([])
  const [scene, setScene] = useState(null)
  const [selected, setSelected] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [seed, setSeed] = useState(4)
  const [size, setSize] = useState(512)
  const [regions, setRegions] = useState([])
  const [region, setRegion] = useState('arabian_sea')
  const [customLL, setCustomLL] = useState('')
  const [drift, setDrift] = useState(null)
  const [attribution, setAttribution] = useState(null)
  const [suspect, setSuspect] = useState(null)
  const [frameIdx, setFrameIdx] = useState(0)
  const [layers, setLayers] = useState({
    sar: true, mask: true, polygons: true, truth: false,
    originHeat: false, driftAnim: false, aisTracks: false,
  })

  useEffect(() => {
    getClasses().then(setClasses).catch(() => {})
    getRegions().then(setRegions).catch(() => {})
    getModules()
      .then((m) => {
        setModules(m)
        // ?seed=N deep-links a specific scene and runs it immediately, so a
        // demo view can be shared as a plain URL.
        const q = new URLSearchParams(window.location.search)
        const s = q.get('seed')
        const r = q.get('region')
        const ll = q.get('at')
        if (r) setRegion(r)
        if (ll) setCustomLL(ll)
        if ((s !== null || r || ll) && m?.detection?.available) {
          if (s !== null) setSeed(s)
          // A shared link should show the whole story, not just detection.
          runFull(s ?? seed, r || region, ll || '')
        }
      })
      .catch((e) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function runDetect(useSeed = seed, useRegion = region, useLL = customLL) {
    setBusy(true); setError(null); setSelected(null)
    try {
      const body = { demo_seed: Number(useSeed), size: Number(size) }
      // "lat, lon" typed by hand wins over the region dropdown, so any patch of
      // ocean on earth can be demoed, not just the presets.
      const m = String(useLL).match(/^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$/)
      if (m) { body.lat = parseFloat(m[1]); body.lon = parseFloat(m[2]) }
      else if (useRegion) { body.region = useRegion }
      const res = await detect(body)
      setScene(res)
    } catch (e) {
      setError(e.message)
      setScene(null)
    } finally {
      setBusy(false)
    }
  }

  async function runFull(useSeed = seed, useRegion = region, useLL = customLL) {
    setBusy(true); setError(null); setSelected(null); setSuspect(null)
    try {
      const body = { demo_seed: Number(useSeed), size: Number(size) }
      const m = String(useLL).match(/^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$/)
      if (m) { body.lat = parseFloat(m[1]); body.lon = parseFloat(m[2]) }
      else if (useRegion) { body.region = useRegion }

      const res = await runPipeline(body)
      setScene(res.detection)
      setDrift(res.drift)
      setAttribution(res.attribution)
      setFrameIdx(0)
      // Turn on the layers the run actually produced, so the result is visible
      // without the user hunting for checkboxes.
      setLayers((l) => ({ ...l, originHeat: !!res.drift,
                          aisTracks: !!res.attribution }))
      if (!res.drift) setError(res.note || 'No slick detected — drift and attribution skipped.')
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const moduleFor = (k) => modules?.[k]
  const detectionReady = moduleFor('detection')?.available

  return (
    <div className="app">
      <div className="topbar">
        <h1>Oil-spill detection &amp; vessel attribution</h1>
        <span className="sub">SIH 26143 · NTRO · Disaster Management</span>
        <span className="spacer" />
        {scene && (
          <span className="sub">
            {scene.source} · {scene.shape[0]}×{scene.shape[1]} px ·
            {' '}{scene.pixel_size_m} m/px · {scene.inference_seconds}s
            {scene.placed_at && ` · ${scene.placed_at.lat.toFixed(2)}, ${scene.placed_at.lon.toFixed(2)}`}
          </span>
        )}
      </div>

      {scene?.georeferencing === 'demo_placement' && (
        <div className="banner">
          Synthetic scene, placed at your chosen location so the map has somewhere
          to draw it — these are not real Sentinel-1 coordinates. Model trained on
          synthetic data; results demonstrate the pipeline, not Sentinel-1
          performance.
        </div>
      )}

      <div className="body">
        {/* ------------------------------------------------ left sidebar -- */}
        <div className="sidebar">
          <ModuleStatus modules={modules} />

          <h2>Run detection</h2>
          {!detectionReady && modules && (
            <div className="err">
              No trained checkpoint. Run:{'\n'}
              python -m detection.train --synthetic --epochs 25
            </div>
          )}
          <label>Region</label>
          <select value={region} disabled={!!customLL}
                  onChange={(e) => setRegion(e.target.value)}>
            {regions.map((r) => (
              <option key={r.id} value={r.id}>{r.name}</option>
            ))}
          </select>

          <label>Or place anywhere (lat, lon)</label>
          <input placeholder="e.g. -33.9, 18.4"
                 value={customLL}
                 onChange={(e) => setCustomLL(e.target.value)} />

          <div className="row">
            <div>
              <label>Scene seed</label>
              <input type="number" value={seed}
                     onChange={(e) => setSeed(e.target.value)} />
            </div>
            <div>
              <label>Size (px)</label>
              <select value={size} onChange={(e) => setSize(e.target.value)}>
                <option value={256}>256</option>
                <option value={512}>512</option>
                <option value={768}>768</option>
              </select>
            </div>
          </div>
          <button disabled={busy || !detectionReady}
                  onClick={() => runFull()}>
            {busy ? 'Running…' : 'Run full pipeline'}
          </button>
          <button className="ghost" disabled={busy || !detectionReady}
                  onClick={() => runDetect()}>
            Detection only
          </button>
          <button className="ghost" disabled={busy || !detectionReady}
                  onClick={() => {
                    const s = Math.floor(Math.random() * 10000)
                    setSeed(s); runFull(s)
                  }}>
            Random scene
          </button>

          {error && <div className="err" style={{ marginTop: 12 }}>{error}</div>}

          <h2>Layers</h2>
          {LAYERS.map((l) => {
            const mod = moduleFor(l.needs)
            const hasData = l.needs === 'drift' ? !!drift
                          : l.needs === 'attribution' ? !!attribution
                          : !!scene
            const unavailable = !mod?.available || !hasData ||
                                (l.synthOnly && !scene?.has_truth)
            const why = !mod?.available
              ? `${l.needs} not built`
              : !hasData ? 'run pipeline'
              : (l.synthOnly && !scene?.has_truth ? 'synthetic only' : '')
            return (
              <label className={`layer ${unavailable ? 'disabled' : ''}`} key={l.key}>
                <input type="checkbox" disabled={unavailable}
                       checked={!!layers[l.key] && !unavailable}
                       onChange={(e) =>
                         setLayers({ ...layers, [l.key]: e.target.checked })} />
                {l.label}
                {why && <span className="why">{why}</span>}
              </label>
            )
          })}

          <h2>Timeline — drift forecast</h2>
          <div className="timeline">
            <input type="range" min="0"
                   max={Math.max((drift?.forecast?.frames?.length || 1) - 1, 0)}
                   value={frameIdx} disabled={!drift}
                   onChange={(e) => setFrameIdx(Number(e.target.value))} />
            <span className="mm" style={{ minWidth: 46, textAlign: 'right' }}>
              {drift ? `+${drift.forecast.frames[frameIdx]?.hours_from_start ?? 0} h` : '—'}
            </span>
          </div>
          <div className="note" style={{ color: 'var(--off)', fontSize: 11 }}>
            {drift
              ? 'Forward drift of the detected slick. Enable "Drift particles" to see it.'
              : 'Run the full pipeline to enable the forecast.'}
          </div>
        </div>

        {/* -------------------------------------------------------- map -- */}
        <div className="mapwrap">
          <MapView scene={scene} layers={layers} selected={selected}
                   onSelect={setSelected} drift={drift}
                   attribution={attribution} frameIdx={frameIdx}
                   selectedSuspect={suspect} onSelectSuspect={setSuspect} />
          {classes.length > 0 && layers.mask && (
            <div className="legend">
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Classes</div>
              {classes.filter((c) => c.name !== 'sea').map((c) => (
                <div className="item" key={c.index}>
                  <span className="swatch" style={{ background: c.color }} />
                  {c.name.replace('_', ' ')}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ----------------------------------------------- right sidebar -- */}
        <div className="rightbar">
          <SlickPanel scene={scene} selected={selected} onSelect={setSelected} />

          {scene && (
            <>
              <h2>Model</h2>
              <div className="stat">
                <span className="k">Architecture</span>
                <span className="v">{scene.model.arch}</span>
              </div>
              <div className="stat">
                <span className="k">Oil IoU (val)</span>
                <span className="v">{scene.model.oil_iou}</span>
              </div>
              <div className="stat">
                <span className="k">Trained on</span>
                <span className="v">{scene.model.trained_on}</span>
              </div>
            </>
          )}

          {drift && (
            <>
              <h2>Drift origin</h2>
              <div className="stat">
                <span className="k">Uncertainty</span>
                <span className="v">{drift.spread_km} km RMS</span>
              </div>
              <div className="stat">
                <span className="k">Ensemble</span>
                <span className="v">{drift.n_particles} members</span>
              </div>
              <div className="stat">
                <span className="k">Forcing</span>
                <span className="v">{drift.forcing.source}</span>
              </div>
              {!drift.forcing.realistic && (
                <div className="uncertain-note">{drift.forcing.note}</div>
              )}
            </>
          )}

          <h2>Suspect vessels</h2>
          <SuspectPanel attribution={attribution} selected={suspect}
                        onSelect={setSuspect} />
        </div>
      </div>
    </div>
  )
}
