// Shows which pipeline modules actually exist. This is deliberately prominent:
// the map has layers for drift and attribution, and a viewer must be able to see
// at a glance that those are unbuilt rather than merely empty.

export default function ModuleStatus({ modules }) {
  if (!modules) return null
  const order = ['screening', 'detection', 'drift', 'attribution']
  return (
    <>
      <h2>Pipeline modules</h2>
      {order.map((key) => {
        const m = modules[key]
        if (!m) return null
        return (
          <div className={`module ${m.available ? '' : 'off'}`} key={key}>
            <span className={`dot ${m.available ? 'on' : 'off'}`} />
            <div>
              <div className="name">
                {m.name} {m.available ? '' : '— not built'}
              </div>
              <div className="note">{m.note}</div>
            </div>
          </div>
        )
      })}
    </>
  )
}
