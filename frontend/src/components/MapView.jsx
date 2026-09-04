import { MapContainer, TileLayer, ImageOverlay, Polygon, CircleMarker,
         Popup, LayersControl, useMap } from 'react-leaflet'
import { useEffect } from 'react'
import { sceneUrl } from '../api'
import { AisTracks, DriftParticles, OriginHeatmap, OriginTrack }
  from './DriftLayers'

// Leaflet wants [lat, lon]; the API emits GeoJSON order [lon, lat].
const toLatLng = ([lon, lat]) => [lat, lon]

/** Pans/zooms to fit everything the run produced, not just the scene.
 *
 * The drift cloud is routinely wider than the SAR footprint, so fitting to the
 * scene alone pushes the origin estimate off-screen — which is the one thing the
 * viewer most needs to see.
 */
function FitBounds({ bounds, drift }) {
  const map = useMap()
  useEffect(() => {
    if (!bounds) return
    let b = [[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]]
    if (drift?.heatmap?.bounds) {
      const [laMin, laMax, loMin, loMax] = drift.heatmap.bounds
      b = [[Math.min(b[0][0], laMin), Math.min(b[0][1], loMin)],
           [Math.max(b[1][0], laMax), Math.max(b[1][1], loMax)]]
    }
    map.fitBounds(b, { padding: [40, 40] })
  }, [bounds, drift, map])
  return null
}

/** Recentres on a slick when one is selected in the sidebar. */
function FlyToSlick({ slick }) {
  const map = useMap()
  useEffect(() => {
    if (slick?.centroid_lonlat) {
      map.flyTo(toLatLng(slick.centroid_lonlat), Math.max(map.getZoom(), 12),
                { duration: 0.6 })
    }
  }, [slick, map])
  return null
}

export default function MapView({ scene, layers, selected, onSelect,
                                  drift, attribution, frameIdx,
                                  selectedSuspect, onSelectSuspect }) {
  const center = scene ? [
    (scene.bounds[0][0] + scene.bounds[1][0]) / 2,
    (scene.bounds[0][1] + scene.bounds[1][1]) / 2,
  ] : [18.53, 71.67]

  return (
    <MapContainer center={center} zoom={10} maxZoom={19} scrollWheelZoom
                  style={{ height: '100%', width: '100%' }}>
      <LayersControl position="topright">
        {/* A scene is only ~15 km across, so the map fits to roughly z13 --
            deeper than some of these services publish. Esri's Ocean basemap has
            no real data over open water past ~z10 and silently serves a "Map
            data not yet available" placeholder instead, so Satellite is the
            default and Ocean is capped to upscale its last real tile. */}
        <LayersControl.BaseLayer checked name="Satellite">
          <TileLayer
            attribution="Esri World Imagery"
            maxNativeZoom={18} maxZoom={19}
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Ocean">
          <TileLayer
            attribution="Esri Ocean"
            maxNativeZoom={10} maxZoom={19}
            url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Plain">
          <TileLayer
            attribution="&copy; OpenStreetMap contributors, &copy; CARTO"
            maxNativeZoom={19} maxZoom={19}
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
        </LayersControl.BaseLayer>
      </LayersControl>

      {scene && <FitBounds bounds={scene.bounds} drift={drift} />}
      {selected && <FlyToSlick slick={selected} />}

      {/* SAR scene */}
      {scene && layers.sar && (
        <ImageOverlay url={sceneUrl(scene.scene_id, 'sar')}
                      bounds={scene.bounds} opacity={0.9} zIndex={200} />
      )}

      {/* Predicted class mask */}
      {scene && layers.mask && (
        <ImageOverlay url={sceneUrl(scene.scene_id, 'mask')}
                      bounds={scene.bounds} opacity={0.65} zIndex={300} />
      )}

      {/* Ground truth, only offered for synthetic scenes */}
      {scene && scene.has_truth && layers.truth && (
        <ImageOverlay url={sceneUrl(scene.scene_id, 'truth')}
                      bounds={scene.bounds} opacity={0.6} zIndex={290} />
      )}

      {/* Drift: origin probability, discharge path, forecast particles */}
      {drift && layers.originHeat && <OriginHeatmap drift={drift} />}
      {drift && layers.originHeat && <OriginTrack drift={drift} />}
      {drift && layers.driftAnim && (
        <DriftParticles drift={drift} frameIdx={frameIdx} />
      )}

      {/* Attribution: AIS tracks coloured by suspicion */}
      {attribution && layers.aisTracks && (
        <AisTracks attribution={attribution} selected={selectedSuspect}
                   onSelect={onSelectSuspect} />
      )}

      {/* Slick polygons + centroids */}
      {scene && layers.polygons && scene.slicks.map((s) => {
        const isSel = selected?.id === s.id
        const poly = (s.polygon_lonlat || []).map(toLatLng)
        return (
          <div key={s.id}>
            {poly.length > 2 && (
              <Polygon
                positions={poly}
                pathOptions={{
                  color: isSel ? '#ffffff' : '#00ffff',
                  weight: isSel ? 3 : 2,
                  fillColor: '#00ffff',
                  fillOpacity: isSel ? 0.35 : 0.18,
                }}
                eventHandlers={{ click: () => onSelect(s) }}
              />
            )}
            {s.centroid_lonlat && (
              <CircleMarker
                center={toLatLng(s.centroid_lonlat)}
                radius={6}
                pathOptions={{ color: '#fff', weight: 2, fillColor: '#00ffff',
                               fillOpacity: 0.9 }}
                eventHandlers={{ click: () => onSelect(s) }}
              >
                <Popup>
                  <b>Slick #{s.id}</b><br />
                  {s.area_km2} km² · axis {s.orientation_deg}°<br />
                  aspect {s.elongation}:1 · {s.contrast_db} dB<br />
                  <span style={{ color: '#8b98a5' }}>
                    age {s.age_saturated === 'upper' ? `> ${s.age_estimate_h}` :
                         s.age_saturated === 'lower' ? `< ${s.age_estimate_h}` :
                         `~${s.age_estimate_h}`} h — low confidence
                  </span>
                </Popup>
              </CircleMarker>
            )}
          </div>
        )
      })}
    </MapContainer>
  )
}
