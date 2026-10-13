import { useCallback, useEffect, useState, type FormEvent } from "react";
import { API, api, apiJson, apiUpload, type Analytics, type Camera, type ImageAnalysis, type Incident, type Page, type Settings, type SystemStatus, formatTime } from "./api";

type Section = "overview" | "monitor" | "sources" | "incidents" | "evidence" | "analytics" | "alerts" | "health" | "settings";
const navigation: { key: Section; label: string; icon: string; group: string }[] = [
  { key: "overview", label: "Overview", icon: "◫", group: "OPERATIONS" },
  { key: "monitor", label: "Live Monitor", icon: "◉", group: "OPERATIONS" },
  { key: "sources", label: "Camera Sources", icon: "⌕", group: "OPERATIONS" },
  { key: "incidents", label: "Incidents & Events", icon: "≡", group: "INVESTIGATE" },
  { key: "evidence", label: "Evidence", icon: "▣", group: "INVESTIGATE" },
  { key: "analytics", label: "Analytics & Reports", icon: "▥", group: "INVESTIGATE" },
  { key: "alerts", label: "Alerts", icon: "!", group: "SYSTEM" },
  { key: "health", label: "System Health", icon: "◇", group: "SYSTEM" },
  { key: "settings", label: "Settings", icon: "⚙", group: "SYSTEM" },
];

const titles: Record<Section, string> = {
  overview: "Operational overview", monitor: "Live monitor", sources: "Camera sources", incidents: "Incidents & events", evidence: "Evidence explorer", analytics: "Analytics & reports", alerts: "Alert queue", health: "System health", settings: "Settings",
};

function App() {
  const [section, setSection] = useState<Section>("overview");
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [events, setEvents] = useState<Page<Incident> | null>(null);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [newAlertCount, setNewAlertCount] = useState(0);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [toast, setToast] = useState("");
  const [query, setQuery] = useState("");
  const [eventType, setEventType] = useState("");
  const [eventStatus, setEventStatus] = useState("");
  const [eventSeverity, setEventSeverity] = useState("");
  const [eventCamera, setEventCamera] = useState("");
  const [eventStart, setEventStart] = useState("");
  const [eventEnd, setEventEnd] = useState("");
  const [sortOrder, setSortOrder] = useState("newest");
  const [page, setPage] = useState(1);
  const [selectedEvent, setSelectedEvent] = useState<Incident | null>(null);
  const [evidenceBroken, setEvidenceBroken] = useState(false);
  const [sourceForm, setSourceForm] = useState({ name: "", source_type: "webcam", source: "0", username: "", password: "" });

  const loadOperational = useCallback(async () => {
    try {
      const [nextCameras, nextSystem, newAlerts] = await Promise.all([
        api<Camera[]>("/api/cameras"),
        api<SystemStatus>("/api/system/status"),
        api<Page<Incident>>("/api/incidents?status=new&page=1&page_size=1"),
      ]);
      setCameras(nextCameras);
      setSystem(nextSystem);
      setNewAlertCount(newAlerts.total);
      setSelectedId((selected) => nextCameras.some((camera) => camera.id === selected) ? selected : nextCameras[0]?.id || "");
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to reach FireSight API");
    }
  }, []);

  const loadEvents = useCallback(async (currentPage = page) => {
    try {
      const params = new URLSearchParams({ page: String(currentPage), page_size: "20", sort: sortOrder });
      if (query) params.set("q", query);
      if (eventType) params.set("event_type", eventType);
      if (eventStatus) params.set("status", eventStatus);
      if (eventSeverity) params.set("severity", eventSeverity);
      if (eventCamera) params.set("camera", eventCamera);
      if (eventStart) params.set("start", eventStart);
      if (eventEnd) params.set("end", eventEnd);
      const endpoint = section === "alerts" ? "/api/alerts" : "/api/incidents";
      const next = await api<Page<Incident>>(`${endpoint}?${params}`);
      setEvents(next);
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to load incidents");
    }
  }, [eventCamera, eventEnd, eventSeverity, eventStart, eventStatus, eventType, page, query, section, sortOrder]);

  useEffect(() => { void loadOperational(); const timer = window.setInterval(loadOperational, 2000); return () => window.clearInterval(timer); }, [loadOperational]);
  useEffect(() => {
    if (["incidents", "evidence", "alerts", "overview"].includes(section)) void loadEvents(section === "overview" ? 1 : page);
  }, [section, page, loadEvents]);
  useEffect(() => {
    if (["analytics", "overview"].includes(section)) void api<Analytics>("/api/analytics/summary").then(setAnalytics).catch((e: Error) => setError(e.message));
    if (section === "settings") void api<Settings>("/api/settings").then(setSettings).catch((e: Error) => setError(e.message));
  }, [section]);
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(""), 3500); return () => window.clearTimeout(timer); }, [toast]);

  const selectedCamera = cameras.find((camera) => camera.id === selectedId) || null;
  const activeCamera = cameras.find((camera) => camera.active && ["connected", "paused"].includes(camera.status)) || null;
  const latestEvents = events?.items.slice(0, 5) || [];
  const doAction = async (key: string, fn: () => Promise<unknown>) => {
    setBusy(key); setError("");
    try { await fn(); setToast("Change saved"); await loadOperational(); if (["incidents", "evidence", "alerts", "overview"].includes(section)) await loadEvents(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Action failed"); }
    finally { setBusy(""); }
  };
  const switchCamera = (id: string) => { setSelectedId(id); setEvidenceBroken(false); };
  const startOverviewSource = async (sourceType: Camera["source_type"], source: string, name: string, username = "", password = "") => {
    setBusy("overview-source"); setError("");
    try {
      const credentials = sourceType === "rtsp" ? { username, password } : {};
      if (sourceType === "rtsp") {
        const tested = await api<{ ok: boolean; diagnostic?: string }>("/api/cameras/test", apiJson("POST", { source, ...credentials }));
        if (!tested.ok) throw new Error(tested.diagnostic || "Camera did not provide a video frame.");
      }
      const camera = await api<Camera>("/api/cameras", apiJson("POST", { name, source_type: sourceType, source, ...credentials }));
      setSelectedId(camera.id);
      await api<Camera>(`/api/cameras/${camera.id}/connect`, apiJson("POST"));
      await loadOperational();
      setToast(`${name} started`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not start source"); }
    finally { setBusy(""); }
  };
  const uploadOverviewVideo = async (file: File, name: string) => {
    setBusy("overview-source"); setError("");
    try {
      const camera = await apiUpload<Camera>(`/api/cameras/upload?name=${encodeURIComponent(name)}`, file);
      setSelectedId(camera.id);
      await api<Camera>(`/api/cameras/${camera.id}/connect`, apiJson("POST"));
      await loadOperational();
      setToast(`${file.name} processing started`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not process video"); }
    finally { setBusy(""); }
  };

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">F</span><span>FireSight <small>AI</small></span></div>
      {(["OPERATIONS", "INVESTIGATE", "SYSTEM"] as const).map((group) => <section className="nav-group" key={group}>
        <div className="nav-caption">{group}</div>
        {navigation.filter((item) => item.group === group).map((item) => <button key={item.key} className={`nav-link ${section === item.key ? "active" : ""}`} onClick={() => setSection(item.key)}><span className="nav-icon">{item.icon}</span><span>{item.label}</span>{item.key === "alerts" && <span className="nav-count">{newAlertCount}</span>}</button>)}
      </section>)}
      <div className="sidebar-footer"><span className={`pulse ${system?.model === "ready" ? "good" : "warn"}`} />{system?.model === "ready" ? "MODEL READY" : system?.model === "error" ? "MODEL ERROR" : "MODEL LOADING"}<small>Local inference · API</small></div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div><h1>{titles[section]}</h1><p>{selectedCamera ? `${selectedCamera.name} · ${selectedCamera.source_label}` : "No camera configured"}</p></div><div className="topbar-right"><span className={`status-pill ${system?.model === "ready" ? "ok" : system?.model === "error" ? "danger" : "warning"}`}><i />{system?.model === "ready" ? "INFERENCE READY" : system?.model === "error" ? "MODEL ERROR" : "INITIALIZING"}</span><time>{new Date().toLocaleTimeString()}</time></div></header>
      {error && <div className="notice error" role="alert"><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}
      <div className="page-content">
        {section === "overview" && <Overview cameras={cameras} system={system} active={activeCamera} selected={selectedCamera} incidents={latestEvents} analytics={analytics} onNavigate={setSection} onSelect={switchCamera} onStartSource={startOverviewSource} onUploadVideo={uploadOverviewVideo} doAction={doAction} busy={busy} />}
        {section === "monitor" && <Monitor cameras={cameras} selected={selectedCamera} selectedId={selectedId} onSelect={switchCamera} doAction={doAction} busy={busy} runImage={(file) => apiUpload<ImageAnalysis>("/api/detect/image", file)} />}
        {section === "sources" && <Sources cameras={cameras} selectedId={selectedId} onSelect={switchCamera} form={sourceForm} setForm={setSourceForm} doAction={doAction} busy={busy} />}
        {(section === "incidents" || section === "alerts") && <IncidentTable events={events} section={section} query={query} setQuery={(value) => { setQuery(value); setPage(1); }} eventType={eventType} setEventType={(value) => { setEventType(value); setPage(1); }} eventStatus={eventStatus} setEventStatus={(value) => { setEventStatus(value); setPage(1); }} eventSeverity={eventSeverity} setEventSeverity={(value) => { setEventSeverity(value); setPage(1); }} eventCamera={eventCamera} setEventCamera={(value) => { setEventCamera(value); setPage(1); }} eventStart={eventStart} setEventStart={(value) => { setEventStart(value); setPage(1); }} eventEnd={eventEnd} setEventEnd={(value) => { setEventEnd(value); setPage(1); }} sortOrder={sortOrder} setSortOrder={(value) => { setSortOrder(value); setPage(1); }} page={page} setPage={setPage} onStatus={(id, status) => doAction(`event-${id}`, () => api(`/api/incidents/${id}`, apiJson("PATCH", { status })))} onEvidence={(event) => { setSelectedEvent(event); setEvidenceBroken(false); setSection("evidence"); }} busy={busy} />}
        {section === "evidence" && <Evidence events={events?.items || []} selected={selectedEvent} onSelect={(event) => { setSelectedEvent(event); setEvidenceBroken(false); }} broken={evidenceBroken} setBroken={setEvidenceBroken} />}
        {section === "analytics" && <AnalyticsView data={analytics} />}
        {section === "health" && <Health cameras={cameras} system={system} onRefresh={() => void loadOperational()} />}
        {section === "settings" && <SettingsView settings={settings} setSettings={setSettings} doAction={doAction} busy={busy} />}
      </div>
      <footer className="footer"><span>FireSight AI · Local security operations</span><span>UTC event timestamps · Browser local time shown</span></footer>
      {toast && <div className="toast" role="status">✓&nbsp; {toast}</div>}
    </main>
  </div>;
}

function SummaryCards({ cameras, system, incidents, analytics }: { cameras: Camera[]; system: SystemStatus | null; incidents: Incident[]; analytics: Analytics | null }) {
  const connected = cameras.filter((camera) => camera.status === "connected" || camera.status === "paused").length;
  const active = cameras.find((camera) => camera.status === "connected" || camera.status === "paused");
  const total = analytics?.total_events ?? incidents.length;
  return <div className="summary-grid">
    <article className="summary-card"><div className="summary-label">CAMERA STATUS</div><div className="summary-main"><strong>{connected} <small>/ {cameras.length}</small></strong><span className={`state-text ${connected ? "teal" : "muted"}`}>{active?.status === "paused" ? "PAUSED" : connected ? "STREAMING" : cameras.length ? "DISCONNECTED" : "NO SOURCES"}</span></div><p>{cameras.length ? `${cameras.filter((camera) => camera.active).length} active · ${cameras.length - cameras.filter((camera) => camera.active).length} idle` : "Add a camera source to begin"}</p></article>
    <article className="summary-card"><div className="summary-label">LIVE PERFORMANCE</div><div className="summary-main"><strong>{active?.fps != null ? active.fps : system?.model === "ready" ? "—" : system?.model === "error" ? "Unavailable" : "Loading"}</strong><span className={`state-text ${active ? "teal" : "muted"}`}>{active ? "FPS" : "NO LIVE SOURCE"}</span></div><p>{active?.inference_ms != null ? `${active.inference_ms} ms inference` : `${system?.active_sources || 0} active · ${system?.max_active_sources || 1} source capacity`}</p></article>
    <article className="summary-card"><div className="summary-label">DETECTION SUMMARY</div><div className="summary-main"><strong>{total}</strong><span className="state-text muted">STORED EVENTS</span></div><p>{analytics?.by_class.fire || 0} fire · {analytics?.by_class.smoke || 0} smoke across persisted history</p></article>
  </div>;
}

function Overview({ cameras, system, active, selected, incidents, analytics, onNavigate, onSelect, onStartSource, onUploadVideo, doAction, busy }: { cameras: Camera[]; system: SystemStatus | null; active: Camera | null; selected: Camera | null; incidents: Incident[]; analytics: Analytics | null; onNavigate: (value: Section) => void; onSelect: (id: string) => void; onStartSource: (type: Camera["source_type"], source: string, name: string, username?: string, password?: string) => Promise<void>; onUploadVideo: (file: File, name: string) => Promise<void>; doAction: (key: string, fn: () => Promise<unknown>) => Promise<void>; busy: string }) {
  const [mode, setMode] = useState<Camera["source_type"]>("video");
  const [device, setDevice] = useState("0");
  const [cameraName, setCameraName] = useState("Laptop Webcam");
  const [rtsp, setRtsp] = useState("");
  const [rtspUsername, setRtspUsername] = useState("");
  const [rtspPassword, setRtspPassword] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [videoName, setVideoName] = useState("");
  const connecting = busy === "overview-source";
  const workerSource = active || (selected?.active ? selected : null);
  const preview = active || (selected?.source_type === "video" && selected.frame_number > 0 ? selected : null);
  const previewUrl = preview ? `${API}/api/cameras/${preview.id}/${preview.active ? "stream" : "snapshot"}` : "";
  const videoComplete = Boolean(preview?.source_type === "video" && !preview.active && preview.video_total_frames && preview.frame_number >= preview.video_total_frames);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (connecting || busy) return;
    if (mode === "video") { if (file) void onUploadVideo(file, videoName.trim() || file.name.replace(/\.[^.]+$/, "")); return; }
    if (mode === "webcam") { void onStartSource("webcam", device, cameraName.trim() || `Webcam ${device}`); return; }
    if (rtsp.trim()) void onStartSource("rtsp", rtsp.trim(), cameraName === "Laptop Webcam" ? "CCTV Camera" : cameraName.trim() || "CCTV Camera", rtspUsername, rtspPassword);
  };
  return <>
    <SummaryCards cameras={cameras} system={system} incidents={incidents} analytics={analytics} />
    <div className="overview-grid">
      <section className="panel preview-panel">
        <div className="panel-heading"><div><h2>Monitoring preview</h2><p>{preview ? `${preview.name} · ${preview.source_type.toUpperCase()} · ${videoComplete ? "complete" : preview.status}` : "No active video source"}</p></div><button className="button subtle" onClick={() => onNavigate("monitor")}>Open monitor&nbsp; ↗</button></div>
        <div className="overview-source-picker">
          <div className="source-mode-tabs" role="tablist" aria-label="Choose video source">
            {([["video", "Video file"], ["webcam", "Laptop webcam"], ["rtsp", "CCTV / IP camera"]] as const).map(([value, label]) => <button type="button" role="tab" aria-selected={mode === value} className={mode === value ? "selected" : ""} key={value} onClick={() => setMode(value)}>{label}</button>)}
          </div>
          <form className="overview-source-form" onSubmit={submit}>
            {mode === "video" && <><label className="field"><span>LOCAL VIDEO · MP4, AVI, MOV, MKV</span><input type="file" accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,.avi,.mov,.mkv,.webm,.wmv" onChange={(event) => { const chosen = event.target.files?.[0] || null; setFile(chosen); setVideoName(chosen?.name.replace(/\.[^.]+$/, "") || ""); }} disabled={!!busy} /></label>{file && <small className="form-hint">{file.name} · uploaded to the local API for validation and processing</small>}</>}
            {mode === "webcam" && <><label className="field"><span>DEVICE INDEX</span><input type="number" min="0" max="32" value={device} onChange={(event) => setDevice(event.target.value)} disabled={!!busy} /></label><label className="field"><span>SOURCE LABEL</span><input value={cameraName} onChange={(event) => setCameraName(event.target.value)} maxLength={80} disabled={!!busy} /></label></>}
            {mode === "rtsp" && <><label className="field"><span>RTSP / RTSPS ENDPOINT</span><input type="password" autoComplete="new-password" spellCheck={false} value={rtsp} onChange={(event) => setRtsp(event.target.value)} placeholder="rtsp://camera-host:554/stream" maxLength={2048} disabled={!!busy} /></label><label className="field"><span>USERNAME · OPTIONAL</span><input autoComplete="off" value={rtspUsername} onChange={(event) => setRtspUsername(event.target.value)} maxLength={256} disabled={!!busy} /></label><label className="field"><span>PASSWORD · OPTIONAL</span><input type="password" autoComplete="new-password" value={rtspPassword} onChange={(event) => setRtspPassword(event.target.value)} maxLength={256} disabled={!!busy} /></label><label className="field"><span>SOURCE LABEL</span><input value={cameraName === "Laptop Webcam" ? "CCTV Camera" : cameraName} onChange={(event) => setCameraName(event.target.value)} maxLength={80} disabled={!!busy} /></label><small className="form-hint">FireSight tests for a readable frame before saving. Credentials and URL parameters are protected with Windows DPAPI and never returned by the API.</small></>}
            <button className="button primary" type="submit" disabled={!!busy || (mode === "video" ? !file : mode === "rtsp" ? !rtsp.trim() : !device.trim())}>{connecting ? "Preparing source…" : mode === "video" ? "Upload & start" : "Connect source"}</button>
          </form>
        </div>
        {cameras.length > 0 && <label className="field overview-active-select"><span>CONFIGURED SOURCE</span><select value={selected?.id || ""} onChange={(event) => onSelect(event.target.value)}>{cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.name} · {camera.status}</option>)}</select></label>}
        {workerSource && <div className="overview-live-controls"><span className={`inline-state ${videoComplete ? "connected" : workerSource.status}`}>{videoComplete ? "COMPLETE" : workerSource.status.toUpperCase()}</span><span>{workerSource.source_label} · frame {workerSource.frame_number}{workerSource.last_frame_at ? ` · ${formatTime(workerSource.last_frame_at)}` : ""}</span><div>
          {workerSource.source_type === "video" && workerSource.status === "connected" && <button className="button subtle" disabled={!!busy} onClick={() => void doAction("pause", () => api(`/api/cameras/${workerSource.id}/pause`, apiJson("POST")))}>Pause</button>}
          {workerSource.source_type === "video" && workerSource.status === "paused" && <button className="button subtle" disabled={!!busy} onClick={() => void doAction("resume", () => api(`/api/cameras/${workerSource.id}/resume`, apiJson("POST")))}>Resume</button>}
          <button className="button danger" disabled={!!busy} onClick={() => void doAction("disconnect", () => api(`/api/cameras/${workerSource.id}/disconnect`, apiJson("POST")))}>Stop</button>
        </div></div>}
        {preview ? <><img className="preview-image" src={previewUrl} alt={`${preview.status === "paused" ? "Paused" : preview.active ? "Live annotated" : "Last processed"} feed from ${preview.name}`} />{preview.source_type === "video" && preview.video_progress != null && <div className="video-progress"><div><span>{videoComplete ? "VIDEO COMPLETE" : preview.status === "paused" ? "VIDEO PAUSED" : "VIDEO PROCESSING"}</span><strong>{Math.round(preview.video_progress * 100)}%</strong></div><progress max="1" value={preview.video_progress} /><small>{preview.frame_number.toLocaleString()} / {(preview.video_total_frames || 0).toLocaleString()} frames{preview.video_duration_seconds ? ` · ${Math.round(preview.video_duration_seconds)} sec source` : ""}</small></div>}</> : <div className="empty-feed"><span className="camera-glyph">◎</span><strong>{selected?.status === "connecting" ? "Connecting to source…" : selected?.status === "error" ? "Source unavailable" : "No live feed"}</strong><p>{selected?.last_error || (selected ? `${selected.name} is idle. Start it here or from Camera Sources.` : "Choose a video file, webcam, or CCTV endpoint to begin.")}</p></div>}
        {selected && !selected.active && <button className="button subtle overview-reconnect" disabled={!!busy || selected.status === "connecting"} onClick={() => void doAction("connect", () => api(`/api/cameras/${selected.id}/connect`, apiJson("POST")))}>Connect selected source</button>}
        {selected?.last_error && <div className="notice error" role="status">{selected.last_error}</div>}
      </section>
      <section className="panel activity-panel"><div className="panel-heading"><div><h2>Recent incidents</h2><p>Latest persisted events</p></div><button className="text-action" onClick={() => onNavigate("incidents")}>View all</button></div>
        {incidents.length ? <div className="activity-list">{incidents.map((event) => <div className="activity-row" key={event.event_id}><span className={`event-marker ${event.event_class}`} /><div className="activity-copy"><strong>{event.event_class} · {(event.confidence * 100).toFixed(1)}%</strong><small>{event.camera_name} · {formatTime(event.timestamp)}</small></div><span className={`severity ${event.severity.toLowerCase()}`}>{event.severity}</span></div>)}</div> : <div className="empty-list">No stored incidents yet.</div>}
        <div className="overview-performance"><strong>Live inference</strong><div><span>FPS</span><b>{active?.fps ?? "—"}</b></div><div><span>Latency</span><b>{active?.inference_ms != null ? `${active.inference_ms} ms` : "—"}</b></div><div><span>Last frame</span><b>{formatTime(active?.last_frame_at)}</b></div><small>Operational metrics · not offline model evaluation</small></div>
      </section>
    </div>
    <section className="panel camera-strip"><div className="panel-heading"><div><h2>Camera availability</h2><p>One source can run inference at a time; state requires a valid captured frame.</p></div><button className="text-action" onClick={() => onNavigate("sources")}>Manage sources</button></div>
      {cameras.length ? <div className="camera-cards">{cameras.slice(0, 4).map((camera) => <button key={camera.id} className="camera-tile" onClick={() => { onSelect(camera.id); onNavigate("monitor"); }}><span className={`online-dot ${camera.status}`} /><span><strong>{camera.name}</strong><small>{camera.source_type.toUpperCase()} · {camera.status}</small></span><span className="tile-chevron">›</span></button>)}</div> : <div className="empty-list">No camera sources configured.</div>}
    </section>
  </>;
}

function Monitor({ cameras, selected, selectedId, onSelect, doAction, busy, runImage }: { cameras: Camera[]; selected: Camera | null; selectedId: string; onSelect: (id: string) => void; doAction: (key: string, fn: () => Promise<unknown>) => Promise<void>; busy: string; runImage: (file: File) => Promise<ImageAnalysis> }) {
  const connected = selected?.status === "connected";
  const [imageAnalysis, setImageAnalysis] = useState<ImageAnalysis | null>(null);
  const [imageError, setImageError] = useState("");
  const [imageBusy, setImageBusy] = useState(false);
  const processImage = async (file?: File) => {
    if (!file) return;
    setImageBusy(true); setImageError(""); setImageAnalysis(null);
    try { setImageAnalysis(await runImage(file)); }
    catch (cause) { setImageError(cause instanceof Error ? cause.message : "Image inference failed"); }
    finally { setImageBusy(false); }
  };
  return <>
    <div className="monitor-toolbar"><label className="field"><span>ACTIVE SOURCE</span><select value={selectedId} onChange={(event) => onSelect(event.target.value)}>{cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.name} · {camera.status}</option>)}</select></label><div className="monitor-stats"><div><span>STATE</span><strong className={`inline-state ${selected?.status || "muted"}`}>{selected?.status || "unavailable"}</strong></div><div><span>FPS</span><strong>{selected?.fps ?? "—"}</strong></div><div><span>INFERENCE</span><strong>{selected?.inference_ms != null ? `${selected.inference_ms} ms` : "—"}</strong></div><div><span>LAST FRAME</span><strong>{selected?.last_frame_at ? formatTime(selected.last_frame_at) : "—"}</strong></div></div><div className="monitor-actions">{selected && (selected.active ? <button className="button danger" disabled={!!busy} onClick={() => void doAction("disconnect", () => api(`/api/cameras/${selected.id}/disconnect`, apiJson("POST")))}>Disconnect</button> : <button className="button primary" disabled={!!busy} onClick={() => void doAction("connect", () => api(`/api/cameras/${selected.id}/connect`, apiJson("POST")))}>{busy === "connect" ? "Connecting…" : "Connect source"}</button>)}</div></div>
    <section className="panel live-canvas">{connected && selected ? <><img key={selected.id} src={`${API}/api/cameras/${selected.id}/stream`} alt={`Live annotated feed from ${selected.name}`} /><span className="live-badge"><i /> LIVE · {selected.name}</span></> : <div className="empty-feed"><span className="camera-glyph">◎</span><strong>{selected?.status === "connecting" ? "Connecting to source…" : selected?.status === "reconnecting" ? "Reconnecting…" : selected?.status === "error" ? "Source error" : "No live feed"}</strong><p>{selected?.last_error || (selected ? "A live feed appears after a valid frame is received." : "Add or select a camera source.")}</p></div>}</section>
    <div className="monitor-bottom"><section className="panel compact-panel"><div className="panel-heading"><div><h2>Source status</h2><p>Truthful capture and inference diagnostics</p></div></div><dl className="details-grid"><dt>Source</dt><dd>{selected?.source_label || "—"}</dd><dt>Frames</dt><dd>{selected?.frame_number ?? "—"}</dd><dt>Last frame</dt><dd>{formatTime(selected?.last_frame_at)}</dd><dt>Last error</dt><dd className={selected?.last_error ? "error-text" : "muted"}>{selected?.last_error || "None"}</dd></dl></section><section className="panel compact-panel"><div className="panel-heading"><div><h2>Analyze an image</h2><p>One real inference pass · max 12 MiB · disabled while live capture is active</p></div><label className="button subtle file-button">{imageBusy ? "Analyzing…" : "Choose image"}<input type="file" accept="image/jpeg,image/png,image/webp" disabled={imageBusy || !!selected?.active} onChange={(event) => { void processImage(event.target.files?.[0]); event.currentTarget.value = ""; }} /></label></div>{imageError && <div className="notice error">{imageError}</div>}{imageAnalysis && <div className="image-result"><img src={`data:image/jpeg;base64,${imageAnalysis.annotated_jpeg_base64}`} alt="YOLO annotated image result" /><div><strong>{imageAnalysis.detections.length} detections · {imageAnalysis.alert_state}</strong><p>{imageAnalysis.detections.map((d) => `${d.class_name} ${(d.confidence * 100).toFixed(1)}%`).join(" · ") || "No detections in this image"}</p><small>{formatTime(imageAnalysis.timestamp)} · {imageAnalysis.confirmed_events.length} confirmed event(s)</small></div></div>}</section></div>
  </>;
}

function Sources({ cameras, selectedId, onSelect, form, setForm, doAction, busy }: { cameras: Camera[]; selectedId: string; onSelect: (id: string) => void; form: { name: string; source_type: string; source: string; username: string; password: string }; setForm: (value: { name: string; source_type: string; source: string; username: string; password: string }) => void; doAction: (key: string, fn: () => Promise<unknown>) => Promise<void>; busy: string }) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [editName, setEditName] = useState("");
  const [editSource, setEditSource] = useState("");
  const [editUsername, setEditUsername] = useState("");
  const [editPassword, setEditPassword] = useState("");
  const selected = cameras.find((camera) => camera.id === selectedId) || null;
  useEffect(() => { setEditName(selected?.name || ""); setEditSource(""); setEditUsername(""); setEditPassword(""); }, [selectedId, selected?.name]);
  const update = (key: string, value: string) => setForm({ ...form, [key]: value });
  return <div className="sources-layout"><section className="panel source-list"><div className="panel-heading"><div><h2>Configured sources</h2><p>Saved source records · credentials are not returned</p></div><span className="count-badge">{cameras.length}</span></div>
    {cameras.length ? cameras.map((camera) => <article className={`source-row ${selectedId === camera.id ? "selected" : ""}`} key={camera.id}><button className="source-select" onClick={() => onSelect(camera.id)}><span className={`online-dot ${camera.status}`} /><span><strong>{camera.name}</strong><small>{camera.source_type.toUpperCase()} · {camera.source_label}</small><small>State: {camera.status} · Last frame: {formatTime(camera.last_frame_at)}</small></span></button><div className="source-actions"><button className="button subtle" disabled={!!busy} onClick={() => void doAction(`test-${camera.id}`, async () => { const result = await api<{ ok: boolean; status: string; diagnostic?: string; width?: number; height?: number }>(`/api/cameras/${camera.id}/test`, apiJson("POST")); if (!result.ok) throw new Error(result.diagnostic || "No valid frame received"); })}>Test</button>{camera.active ? <button className="button danger" disabled={!!busy} onClick={() => void doAction(`stop-${camera.id}`, () => api(`/api/cameras/${camera.id}/disconnect`, apiJson("POST")))}>Stop</button> : <button className="button primary" disabled={!!busy} onClick={() => void doAction(`start-${camera.id}`, () => api(`/api/cameras/${camera.id}/connect`, apiJson("POST")))}>Connect</button>}<button className="icon-button" aria-label={`Remove ${camera.name}`} disabled={!!busy || camera.active} onClick={() => void doAction(`delete-${camera.id}`, () => api(`/api/cameras/${camera.id}`, { method: "DELETE" }))}>×</button></div></article>) : <div className="empty-list">No sources configured.</div>}
    {selected && <form className="edit-source-form" onSubmit={(event) => { event.preventDefault(); void doAction(`edit-${selected.id}`, async () => { const hasRtspChanges = Boolean(editSource.trim() || editUsername || editPassword); const rtspCredentials = { username: editUsername, password: editPassword }; if (selected.source_type === "rtsp" && hasRtspChanges) { const tested = await api<{ ok: boolean; diagnostic?: string }>(`/api/cameras/${selected.id}/test`, apiJson("POST", { ...(editSource.trim() ? { source: editSource.trim() } : {}), ...rtspCredentials })); if (!tested.ok) throw new Error(tested.diagnostic || "Camera did not provide a video frame."); } await api(`/api/cameras/${selected.id}`, apiJson("PATCH", { name: editName, ...(editSource.trim() ? { source: editSource.trim() } : {}), ...(selected.source_type === "rtsp" && hasRtspChanges ? rtspCredentials : {}) })); }); }}><div className="panel-heading"><div><h2>Edit selected source</h2><p>Credentials and saved endpoints are not returned to the browser.</p></div></div><label className="field"><span>DISPLAY NAME</span><input value={editName} onChange={(event) => setEditName(event.target.value)} required maxLength={80} /></label><label className="field"><span>REPLACE SOURCE (OPTIONAL)</span><input value={editSource} onChange={(event) => setEditSource(event.target.value)} type={selected.source_type === "rtsp" ? "password" : "text"} autoComplete="off" placeholder={selected.source_type === "rtsp" ? "rtsp://camera/stream" : selected.source_type === "video" ? "Server-local path to a video" : "Webcam device index"} /></label>{selected.source_type === "rtsp" && <><label className="field"><span>NEW USERNAME · OPTIONAL</span><input autoComplete="off" value={editUsername} onChange={(event) => setEditUsername(event.target.value)} maxLength={256} /></label><label className="field"><span>NEW PASSWORD · OPTIONAL</span><input type="password" autoComplete="new-password" value={editPassword} onChange={(event) => setEditPassword(event.target.value)} maxLength={256} /></label><small>Entered changes are tested before save. Leave the endpoint blank to retain it; blank credential fields retain embedded URL credentials.</small></>}<small>Leave blank to retain the current source. Video uploads can be replaced by adding a new upload and removing this entry.</small><button className="button subtle" disabled={!!busy}>Save source changes</button></form>}
  </section><section className="panel add-source"><div className="panel-heading"><div><h2>Add a source</h2><p>Source configuration is saved locally.</p></div></div><form onSubmit={(event) => { event.preventDefault(); void doAction("add-source", async () => { if (form.source_type === "video") { if (!videoFile) throw new Error("Choose a local video file first"); const params = new URLSearchParams({ name: form.name }); await apiUpload(`/api/cameras/upload?${params}`, videoFile); setVideoFile(null); } else { if (form.source_type === "rtsp") { const tested = await api<{ ok: boolean; diagnostic?: string }>("/api/cameras/test", apiJson("POST", { source: form.source, username: form.username, password: form.password })); if (!tested.ok) throw new Error(tested.diagnostic || "Camera did not provide a video frame."); } await api("/api/cameras", apiJson("POST", { name: form.name, source_type: form.source_type, source: form.source, ...(form.source_type === "rtsp" ? { username: form.username, password: form.password } : {}) })); } setForm({ name: "", source_type: "webcam", source: "0", username: "", password: "" }); }); }}>
    <label className="field"><span>DISPLAY NAME</span><input required maxLength={80} value={form.name} onChange={(event) => update("name", event.target.value)} placeholder="North entrance" /></label>
    <label className="field"><span>SOURCE TYPE</span><select value={form.source_type} onChange={(event) => setForm({ ...form, source_type: event.target.value, source: event.target.value === "webcam" ? "0" : "", username: "", password: "" })}><option value="webcam">Webcam / device index</option><option value="rtsp">RTSP / RTSPS stream</option><option value="video">Local video path</option></select></label>
    {form.source_type === "video" ? <label className="field"><span>LOCAL VIDEO FILE · MAX 200 MIB</span><input required type="file" accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm,video/x-msvideo" onChange={(event) => setVideoFile(event.target.files?.[0] || null)} /><small>The API verifies a readable frame before saving the uploaded clip.</small></label> : <label className="field"><span>{form.source_type === "webcam" ? "DEVICE INDEX" : "RTSP ENDPOINT"}</span><input required value={form.source} onChange={(event) => update("source", event.target.value)} placeholder={form.source_type === "webcam" ? "0" : "rtsp://camera/stream"} type={form.source_type === "rtsp" ? "password" : "text"} autoComplete="off" /></label>}
    {form.source_type === "rtsp" && <><label className="field"><span>USERNAME · OPTIONAL</span><input autoComplete="off" value={form.username} onChange={(event) => update("username", event.target.value)} maxLength={256} /></label><label className="field"><span>PASSWORD · OPTIONAL</span><input type="password" autoComplete="new-password" value={form.password} onChange={(event) => update("password", event.target.value)} maxLength={256} /></label><p className="form-hint">The endpoint is tested for a readable frame before saving. The complete URL, credentials, and query parameters are protected with Windows DPAPI.</p></>}
    <button className="button primary full" disabled={!!busy || form.source_type === "video" && !videoFile}>{busy === "add-source" ? "Saving…" : "Add camera source"}</button>
  </form></section></div>;
}

function IncidentTable({ events, section, query, setQuery, eventType, setEventType, eventStatus, setEventStatus, eventSeverity, setEventSeverity, eventCamera, setEventCamera, eventStart, setEventStart, eventEnd, setEventEnd, sortOrder, setSortOrder, page, setPage, onStatus, onEvidence, busy }: { events: Page<Incident> | null; section: "incidents" | "alerts"; query: string; setQuery: (value: string) => void; eventType: string; setEventType: (value: string) => void; eventStatus: string; setEventStatus: (value: string) => void; eventSeverity: string; setEventSeverity: (value: string) => void; eventCamera: string; setEventCamera: (value: string) => void; eventStart: string; setEventStart: (value: string) => void; eventEnd: string; setEventEnd: (value: string) => void; sortOrder: string; setSortOrder: (value: string) => void; page: number; setPage: (value: number) => void; onStatus: (id: string, status: string) => void; onEvidence: (event: Incident) => void; busy: string }) {
  const items = events?.items || [];
  return <section className="panel table-panel"><div className="panel-heading"><div><h2>{section === "alerts" ? "Operational alert queue" : "Confirmed incidents"}</h2><p>Persisted events from the existing confirmation pipeline</p></div><span className="count-badge">{events?.total ?? "—"}</span></div><div className="filters"><label className="search-field"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search event, camera, or ID" /></label><select aria-label="Filter event type" value={eventType} onChange={(event) => setEventType(event.target.value)}><option value="">All event types</option><option value="fire">Fire</option><option value="smoke">Smoke</option><option value="person">Person</option></select><select aria-label="Filter severity" value={eventSeverity} onChange={(event) => setEventSeverity(event.target.value)}><option value="">All severities</option><option value="critical">Critical</option><option value="warning">Warning</option><option value="info">Info</option></select><select aria-label="Filter incident status" value={eventStatus} onChange={(event) => setEventStatus(event.target.value)}><option value="">{section === "alerts" ? "All open alerts" : "All statuses"}</option><option value="new">New</option><option value="acknowledged">Acknowledged</option><option value="investigating">Investigating</option>{section !== "alerts" && <option value="resolved">Resolved</option>}</select><input aria-label="Filter source name" className="filter-input" value={eventCamera} onChange={(event) => setEventCamera(event.target.value)} placeholder="Camera / source" /><label className="date-filter"><span>FROM</span><input aria-label="Start date" type="date" value={eventStart} onChange={(event) => setEventStart(event.target.value)} /></label><label className="date-filter"><span>TO</span><input aria-label="End date" type="date" value={eventEnd} onChange={(event) => setEventEnd(event.target.value)} /></label><select aria-label="Sort incidents" value={sortOrder} onChange={(event) => setSortOrder(event.target.value)}><option value="newest">Newest first</option><option value="oldest">Oldest first</option></select></div>
    <div className="table-wrap"><table><thead><tr><th>TIME</th><th>SOURCE</th><th>EVENT</th><th>CONF.</th><th>SEVERITY</th><th>STATUS</th><th>ACTIONS</th></tr></thead><tbody>{items.map((event) => <tr key={event.event_id}><td className="time-cell">{formatTime(event.timestamp)}</td><td>{event.camera_name}</td><td><span className={`event-type ${event.event_class}`}>{event.event_class}</span></td><td>{(event.confidence * 100).toFixed(1)}%</td><td><span className={`severity ${event.severity.toLowerCase()}`}>{event.severity}</span></td><td><span className={`status-tag ${event.status}`}>{event.status}</span></td><td><div className="row-actions"><button className="text-action" onClick={() => onEvidence(event)}>Evidence</button>{event.status !== "resolved" && <select aria-label={`Update ${event.event_id} status`} disabled={busy === `event-${event.event_id}`} value={event.status} onChange={(e) => onStatus(event.event_id, e.target.value)}><option value="new">New</option><option value="acknowledged">Acknowledge</option><option value="investigating">Investigating</option><option value="resolved">Resolve</option></select>}</div></td></tr>)}</tbody></table>{!items.length && <div className="empty-list">{events ? "No matching events in the stored history." : "Loading persisted events…"}</div>}</div>
    <div className="pagination"><span>{events ? `${events.total} records · Page ${events.page}` : "—"}</span><div><button className="button subtle" disabled={page <= 1} onClick={() => setPage(page - 1)}>Previous</button><button className="button subtle" disabled={!events?.has_more} onClick={() => setPage(page + 1)}>Next</button></div></div>
  </section>;
}

function Evidence({ events, selected, onSelect, broken, setBroken }: { events: Incident[]; selected: Incident | null; onSelect: (event: Incident) => void; broken: boolean; setBroken: (value: boolean) => void }) {
  const current = selected || events[0] || null;
  return <div className="evidence-layout"><section className="panel evidence-list"><div className="panel-heading"><div><h2>Event evidence</h2><p>Evidence referenced by persisted events</p></div></div>{events.map((event) => <button className={`evidence-item ${current?.event_id === event.event_id ? "selected" : ""}`} key={event.event_id} onClick={() => onSelect(event)}><span className={`event-marker ${event.event_class}`} /><span><strong>{event.event_class} · {event.camera_name}</strong><small>{formatTime(event.timestamp)} · {(event.confidence * 100).toFixed(1)}%</small></span></button>)}{!events.length && <div className="empty-list">No evidence records are available.</div>}</section>
    <section className="panel evidence-detail"><div className="panel-heading"><div><h2>Evidence detail</h2><p>{current ? current.event_id : "Choose an event to inspect"}</p></div>{current && <a className="button subtle" href={`${API}/api/evidence/${encodeURIComponent(current.event_id)}?download=true`} download>Download original</a>}</div>{current ? <><div className="evidence-preview">{broken ? <div className="empty-feed"><strong>Evidence unavailable</strong><p>The file may have been removed or is unreadable.</p></div> : <img src={`${API}/api/evidence/${encodeURIComponent(current.event_id)}`} alt={`Evidence for ${current.event_class} event`} onError={() => setBroken(true)} />}</div><dl className="details-grid"><dt>Event type</dt><dd>{current.event_class}</dd><dt>Confidence</dt><dd>{(current.confidence * 100).toFixed(1)}% · model score, not calibrated risk</dd><dt>Severity</dt><dd>{current.severity}</dd><dt>State</dt><dd>{current.status}</dd><dt>Source</dt><dd>{current.camera_name}</dd><dt>Capture time</dt><dd>{formatTime(current.timestamp)}</dd></dl></> : <div className="empty-feed"><strong>No event selected</strong><p>Events with recorded evidence will appear here.</p></div>}</section></div>;
}

function AnalyticsView({ data }: { data: Analytics | null }) {
  if (!data) return <div className="loading-panel">Loading analytics from stored events…</div>;
  const max = Math.max(1, ...data.timeseries.map((row) => Object.values(row.counts).reduce((sum, value) => sum + value, 0)));
  return <><div className="metric-grid"><Metric label="TOTAL EVENTS" value={data.total_events} note="All persisted events" /><Metric label="ACKNOWLEDGED" value={data.acknowledged} note="Includes investigating and resolved" /><Metric label="RESOLVED" value={data.resolved} note="Stored lifecycle state" /><Metric label="CAMERAS" value={data.cameras.length} note="Configured sources" /></div><section className="panel analytics-panel"><div className="panel-heading"><div><h2>Event activity</h2><p>Confirmed detections · trailing {data.days} days · {data.timezone}</p></div><a className="button subtle" href={`${API}/api/analytics/export.csv`}>Export CSV</a></div>{data.timeseries.length ? <div className="chart-list">{data.timeseries.map((row) => { const sum = Object.values(row.counts).reduce((a, b) => a + b, 0); return <div className="chart-row" key={row.date}><time>{row.date}</time><div className="bar-track"><span style={{ width: `${Math.max(2, sum / max * 100)}%` }} /></div><strong>{sum}</strong><small>{Object.entries(row.counts).map(([key, value]) => `${key} ${value}`).join(" · ")}</small></div>; })}</div> : <div className="empty-list">No persisted events in this time range.</div>}</section><div className="overview-grid"><section className="panel compact-panel"><div className="panel-heading"><div><h2>Event types</h2><p>All persisted event records</p></div></div><dl className="details-grid">{Object.entries(data.by_class).length ? Object.entries(data.by_class).map(([key, value]) => <FragmentRow key={key} label={key} value={String(value)} />) : <dd className="muted">No events stored</dd>}</dl></section><section className="panel compact-panel"><div className="panel-heading"><div><h2>Camera state</h2><p>Current operational status</p></div></div>{data.cameras.map((camera) => <div className="activity-row" key={camera.id}><span className={`online-dot ${camera.status}`} /><div className="activity-copy"><strong>{camera.name}</strong><small>{camera.source_type.toUpperCase()} · {camera.status}</small></div></div>)}</section></div></>;
}

function Metric({ label, value, note }: { label: string; value: string | number; note: string }) { return <article className="summary-card"><div className="summary-label">{label}</div><div className="summary-main"><strong>{value}</strong></div><p>{note}</p></article>; }
function FragmentRow({ label, value }: { label: string; value: string }) { return <><dt>{label}</dt><dd>{value}</dd></>; }

function Health({ cameras, system, onRefresh }: { cameras: Camera[]; system: SystemStatus | null; onRefresh: () => void }) {
  return <><div className="health-header"><div><h2>Operational diagnostics</h2><p>Process health and model readiness are reported independently.</p></div><button className="button subtle" onClick={onRefresh}>Refresh</button></div><div className="metric-grid"><Metric label="API PROCESS" value={system?.process || "Unavailable"} note={`Uptime ${system ? Math.floor(system.uptime_seconds / 60) : 0} min`} /><Metric label="MODEL" value={system?.model || "Unknown"} note={system?.model_error || "Readiness from model loader"} /><Metric label="ACTIVE SOURCES" value={`${system?.active_sources ?? 0} / ${system?.max_active_sources ?? 1}`} note="Configured active capacity" /><Metric label="EVIDENCE STORE" value={system?.storage.evidence_available ? "Available" : "Unavailable"} note={`Database ${system?.storage.database_available ? "available" : "unavailable"}`} /></div><section className="panel health-panel"><div className="panel-heading"><div><h2>Source health</h2><p>Connection succeeds only after an actual valid frame.</p></div></div>{cameras.length ? cameras.map((camera) => <div className="health-row" key={camera.id}><span className={`online-dot ${camera.status}`} /><div><strong>{camera.name}</strong><small>{camera.source_type.toUpperCase()} · {camera.status} · frame {camera.frame_number}</small><small>Last frame: {formatTime(camera.last_frame_at)}</small>{camera.last_error && <small className="error-text">{camera.last_error}</small>}</div><div className="health-metric">{camera.fps ?? "—"}<small>FPS</small></div><div className="health-metric">{camera.inference_ms != null ? `${camera.inference_ms} ms` : "—"}<small>INFERENCE</small></div></div>) : <div className="empty-list">No cameras configured.</div>}</section><aside className="notice warning"><span>This development API binds to loopback and has no authentication. Do not expose it to other machines; add deployment authentication and TLS before network access.</span></aside></>;
}

function SettingsView({ settings, setSettings, doAction, busy }: { settings: Settings | null; setSettings: (value: Settings | null) => void; doAction: (key: string, fn: () => Promise<unknown>) => Promise<void>; busy: string }) {
  if (!settings) return <div className="loading-panel">Loading validated settings…</div>;
  const save = () => void doAction("settings", async () => { const saved = await api<Settings>("/api/settings", apiJson("PATCH", settings)); setSettings(saved); });
  return <section className="panel settings-panel"><div className="panel-heading"><div><h2>Inference & reconnect policy</h2><p>Validated and persisted separately from the existing desktop configuration.</p></div></div><div className="settings-grid"><label className="field"><span>CONFIDENCE THRESHOLD</span><input type="number" min="0" max="1" step="0.01" value={settings.confidence_threshold} onChange={(e) => setSettings({ ...settings, confidence_threshold: Number(e.target.value) })} /><small>Existing YOLO score cutoff; not a calibrated danger probability.</small></label><label className="field"><span>CONFIRMATION FRAMES</span><input type="number" min="1" max="60" step="1" value={settings.confirmation_frames} onChange={(e) => setSettings({ ...settings, confirmation_frames: Number(e.target.value) })} /><small>Consecutive frames required by the existing confirmer.</small></label><label className="field"><span>RTSP RECONNECT ATTEMPTS</span><input type="number" min="0" max="10" step="1" value={settings.reconnect_attempts} onChange={(e) => setSettings({ ...settings, reconnect_attempts: Number(e.target.value) })} /><small>Bounded retries with incremental backoff.</small></label><label className="field"><span>MAX ACTIVE SOURCES</span><input value={settings.max_active_sources} disabled /><small>One active source: the existing event confirmer and model worker are shared.</small></label></div><button className="button primary" disabled={!!busy} onClick={save}>{busy === "settings" ? "Saving…" : "Save settings"}</button></section>;
}

export default App;
