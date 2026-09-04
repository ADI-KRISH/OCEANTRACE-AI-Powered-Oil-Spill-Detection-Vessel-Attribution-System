// Thin client for the FastAPI backend. Every call returns parsed JSON or throws
// with the server's own detail message, so the UI can surface real errors
// instead of a generic failure.

async function req(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  let body = null
  try {
    body = await res.json()
  } catch {
    /* non-JSON (e.g. an image route hit by mistake) */
  }
  if (!res.ok) {
    const msg = body?.detail || body?.error || `${res.status} ${res.statusText}`
    throw new Error(msg)
  }
  return body
}

export const getModules = () => req('/api/modules')
export const getClasses = () => req('/api/classes')
export const getRegions = () => req('/api/regions')
export const detect = (payload) =>
  req('/api/detect', { method: 'POST', body: JSON.stringify(payload) })

export const runPipeline = (payload) =>
  req('/api/pipeline/run', { method: 'POST', body: JSON.stringify(payload) })

export const sceneUrl = (sceneId, layer) => `/api/scene/${sceneId}/${layer}.png`
