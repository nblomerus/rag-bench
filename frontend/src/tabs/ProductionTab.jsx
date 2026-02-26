import React, { useState, useEffect, useRef } from 'react'
import { fetchStats, fetchBenchmarkHistory, fetchMetricsSummary } from '../utils/api'
import { CpuIcon, BookIcon, DatabaseIcon, ZapIcon, SearchIcon, BarChartIcon, Spinner } from '../components/Icons'

const MAX_HW_HISTORY = 4320 // 12h at 10s intervals

const TIME_RANGES = [
    { key: '30m', label: '30m', ms: 30 * 60 * 1000 },
    { key: '1h',  label: '1h',  ms: 60 * 60 * 1000 },
    { key: '3h',  label: '3h',  ms: 3 * 60 * 60 * 1000 },
    { key: '6h',  label: '6h',  ms: 6 * 60 * 60 * 1000 },
    { key: '12h', label: '12h', ms: 12 * 60 * 60 * 1000 },
]

function filterByTime(data, rangeMs) {
    if (!data || data.length === 0) return data
    const cutoff = Date.now() - rangeMs
    return data.filter(d => d.t && new Date(d.t).getTime() >= cutoff)
}

function formatUptime(seconds) {
    if (!seconds) return '—'
    const d = Math.floor(seconds / 86400)
    const h = Math.floor((seconds % 86400) / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    if (d > 0) return `${d}d ${h}h ${m}m`
    if (h > 0) return `${h}h ${m}m`
    return `${m}m`
}

// ── Grafana-style SVG time-series chart ──

function TimeChart({ data, series, title, yFormat, height = 180, emptyMessage, rangeMs }) {
    const [hoverIdx, setHoverIdx] = useState(null)
    const svgRef = useRef(null)
    const cid = useRef(`tc${Math.random().toString(36).slice(2, 8)}`).current

    if (!data || data.length < 2) {
        if (!emptyMessage) return null
        return (
            <div className="mt-3">
                <div className="rounded-lg overflow-hidden" style={{ background: 'var(--apple-chart-bg)', border: '1px solid var(--apple-chart-border)' }}>
                    <div className="px-3 pt-2.5 pb-1">
                        <span className="text-[11px] font-medium" style={{ color: 'var(--apple-chart-label)' }}>{title}</span>
                    </div>
                    <div className="flex items-center justify-center" style={{ height, background: 'var(--apple-chart-canvas)' }}>
                        <span className="text-xs" style={{ color: 'var(--apple-text-quaternary)' }}>{emptyMessage}</span>
                    </div>
                    <div className="flex items-center gap-4 px-3 py-1.5 justify-center flex-wrap" style={{ borderTop: '1px solid var(--apple-chart-grid)' }}>
                        {series.map(s => (
                            <span key={s.label} className="flex items-center gap-1.5 text-[10px]" style={{ color: 'var(--apple-chart-axis)' }}>
                                <span className="inline-block w-3 h-0.5 rounded" style={{ backgroundColor: s.color, opacity: 0.4 }} />
                                {s.label}
                            </span>
                        ))}
                    </div>
                </div>
            </div>
        )
    }

    const W = 600, H = height, PAD_L = 52, PAD_R = 12, PAD_T = 8, PAD_B = 24
    const chartW = W - PAD_L - PAD_R
    const chartH = H - PAD_T - PAD_B

    // Compute max across all series
    let maxVal = 0
    for (const s of series) {
        for (const d of data) {
            const v = typeof s.key === 'function' ? s.key(d) : d[s.key]
            if (v != null && v > maxVal) maxVal = v
        }
    }
    if (maxVal === 0) maxVal = 1

    // Nice y-axis tick calculation
    const rawStep = maxVal / 4
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep || 1)))
    const niceStep = Math.ceil(rawStep / mag) * mag || 1
    const yTicks = []
    for (let v = 0; v <= maxVal + niceStep * 0.5; v += niceStep) yTicks.push(Math.round(v * 100) / 100)
    const yMax = yTicks[yTicks.length - 1] || maxVal * 1.15

    // Timestamp-based x positioning anchored to [tMin, tMax]
    const tMax = Date.now()
    const tMin = rangeMs ? tMax - rangeMs : (data[0]?.t ? new Date(data[0].t).getTime() : tMax - 60 * 60 * 1000)
    const tSpan = tMax - tMin || 1

    function xPos(i) {
        const t = data[i]?.t ? new Date(data[i].t).getTime() : tMin
        return PAD_L + ((t - tMin) / tSpan) * chartW
    }
    function yPos(v) { return PAD_T + chartH - (Math.min(v, yMax) / yMax) * chartH }
    const yBottom = PAD_T + chartH

    // Catmull-Rom spline → cubic Bezier SVG path (smooth curves)
    function smoothPath(s) {
        const pts = data.map((d, i) => {
            const v = typeof s.key === 'function' ? s.key(d) : d[s.key]
            return { x: xPos(i), y: yPos(v || 0) }
        })
        if (pts.length === 2) return `M${pts[0].x},${pts[0].y}L${pts[1].x},${pts[1].y}`
        let path = `M${pts[0].x},${pts[0].y}`
        for (let i = 0; i < pts.length - 1; i++) {
            const p0 = pts[Math.max(i - 1, 0)]
            const p1 = pts[i]
            const p2 = pts[i + 1]
            const p3 = pts[Math.min(i + 2, pts.length - 1)]
            path += `C${p1.x + (p2.x - p0.x) / 6},${p1.y + (p2.y - p0.y) / 6},${p2.x - (p3.x - p1.x) / 6},${p2.y - (p3.y - p1.y) / 6},${p2.x},${p2.y}`
        }
        return path
    }

    function fillPath(s) {
        return `${smoothPath(s)}L${xPos(data.length - 1)},${yBottom}L${xPos(0)},${yBottom}Z`
    }

    const fmtY = yFormat || (v => v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`)

    // X-axis time labels (evenly spaced across the time range, like Grafana)
    const xLabelCount = 5
    const xTickTimes = Array.from({ length: xLabelCount }, (_, i) =>
        tMin + (i / (xLabelCount - 1)) * tSpan
    )

    function handleMouseMove(e) {
        const svg = svgRef.current
        if (!svg) return
        const rect = svg.getBoundingClientRect()
        const mouseX = ((e.clientX - rect.left) / rect.width) * W
        let nearest = 0, minDist = Infinity
        for (let i = 0; i < data.length; i++) {
            const d = Math.abs(xPos(i) - mouseX)
            if (d < minDist) { minDist = d; nearest = i }
        }
        setHoverIdx(nearest)
    }

    // Tooltip geometry
    const tipW = 140, tipRowH = 16, tipPadY = 22
    const tipH = hoverIdx != null ? tipPadY + series.length * tipRowH + 4 : 0
    const tipRawX = hoverIdx != null ? xPos(hoverIdx) + 12 : 0
    const tipX = tipRawX + tipW > W - PAD_R ? xPos(hoverIdx) - tipW - 12 : tipRawX
    const tipY = PAD_T + 4

    return (
        <div className="mt-3">
            <div className="rounded-lg overflow-hidden" style={{ background: 'var(--apple-chart-bg)', border: '1px solid var(--apple-chart-border)' }}>
                {/* Panel header */}
                <div className="px-3 pt-2.5 pb-1">
                    <span className="text-[11px] font-medium" style={{ color: 'var(--apple-chart-label)' }}>{title}</span>
                </div>
                {/* Chart SVG */}
                <svg
                    ref={svgRef}
                    viewBox={`0 0 ${W} ${H}`}
                    className="w-full"
                    style={{ background: 'var(--apple-chart-canvas)', display: 'block' }}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={() => setHoverIdx(null)}
                >
                    <defs>
                        {series.map((s, idx) => s.fill && (
                            <linearGradient key={idx} id={`${cid}g${idx}`} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor={s.color} stopOpacity="0.3" />
                                <stop offset="100%" stopColor={s.color} stopOpacity="0.03" />
                            </linearGradient>
                        ))}
                    </defs>

                    {/* Horizontal grid lines (dashed) */}
                    {yTicks.map((v, i) => (
                        <g key={i}>
                            <line x1={PAD_L} y1={yPos(v)} x2={W - PAD_R} y2={yPos(v)}
                                style={{ stroke: 'var(--apple-chart-grid)' }} strokeWidth="1" strokeDasharray={v === 0 ? undefined : '4,3'} />
                            <text x={PAD_L - 8} y={yPos(v) + 3} textAnchor="end"
                                style={{ fill: 'var(--apple-chart-axis)' }} fontSize="9" fontFamily="ui-monospace,monospace">{fmtY(v)}</text>
                        </g>
                    ))}

                    {/* Gradient fills */}
                    {series.map((s, idx) => s.fill && (
                        <path key={`f${idx}`} d={fillPath(s)} fill={`url(#${cid}g${idx})`} />
                    ))}

                    {/* Smooth data lines */}
                    {series.map((s, idx) => (
                        <path key={`l${idx}`} d={smoothPath(s)}
                            fill="none" stroke={s.color} strokeWidth="1.5"
                            strokeLinejoin="round" strokeLinecap="round" />
                    ))}

                    {/* Hover crosshair + tooltip */}
                    {hoverIdx != null && (
                        <>
                            <line x1={xPos(hoverIdx)} y1={PAD_T} x2={xPos(hoverIdx)} y2={yBottom}
                                style={{ stroke: 'var(--apple-chart-axis)' }} strokeWidth="1" strokeDasharray="3,2" />
                            {series.map((s, idx) => {
                                const v = typeof s.key === 'function' ? s.key(data[hoverIdx]) : data[hoverIdx][s.key]
                                return <circle key={idx} cx={xPos(hoverIdx)} cy={yPos(v || 0)} r="3.5"
                                    fill={s.color} style={{ stroke: 'var(--apple-chart-canvas)' }} strokeWidth="1.5" />
                            })}
                            <rect x={tipX} y={tipY} width={tipW} height={tipH}
                                rx="4" style={{ fill: 'var(--apple-chart-bg)', stroke: 'var(--apple-chart-border)' }} fillOpacity="0.95" strokeWidth="1" />
                            <text x={tipX + 8} y={tipY + 14} style={{ fill: 'var(--apple-chart-label)' }} fontSize="9" fontFamily="ui-monospace,monospace">
                                {data[hoverIdx].t ? new Date(data[hoverIdx].t).toLocaleTimeString() : ''}
                            </text>
                            {series.map((s, i) => {
                                const v = typeof s.key === 'function' ? s.key(data[hoverIdx]) : data[hoverIdx][s.key]
                                return (
                                    <g key={i}>
                                        <circle cx={tipX + 12} cy={tipY + tipPadY + 6 + i * tipRowH} r="3" fill={s.color} />
                                        <text x={tipX + 20} y={tipY + tipPadY + 10 + i * tipRowH}
                                            style={{ fill: 'var(--apple-chart-label)' }} fontSize="9">{s.label}</text>
                                        <text x={tipX + tipW - 8} y={tipY + tipPadY + 10 + i * tipRowH}
                                            fill={s.color} fontSize="9" fontFamily="ui-monospace,monospace" textAnchor="end">
                                            {fmtY(v || 0)}
                                        </text>
                                    </g>
                                )
                            })}
                        </>
                    )}

                    {/* X-axis time labels */}
                    {xTickTimes.map((tickMs, i) => {
                        const t = new Date(tickMs)
                        const tickX = PAD_L + (i / (xLabelCount - 1)) * chartW
                        return (
                            <text key={i} x={tickX} y={H - 4} textAnchor="middle"
                                style={{ fill: 'var(--apple-chart-axis)' }} fontSize="9" fontFamily="ui-monospace,monospace">
                                {t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </text>
                        )
                    })}
                </svg>

                {/* Legend with current values */}
                <div className="flex items-center gap-4 px-3 py-1.5 justify-center flex-wrap" style={{ borderTop: '1px solid var(--apple-chart-grid)' }}>
                    {series.map(s => {
                        const idx = hoverIdx != null ? hoverIdx : data.length - 1
                        const val = typeof s.key === 'function' ? s.key(data[idx]) : data[idx][s.key]
                        return (
                            <span key={s.label} className="flex items-center gap-1.5 text-[10px]" style={{ color: 'var(--apple-chart-label)' }}>
                                <span className="inline-block w-3 h-0.5 rounded" style={{ backgroundColor: s.color }} />
                                {s.label}
                                <span className="font-mono" style={{ color: s.color }}>{fmtY(val || 0)}</span>
                            </span>
                        )
                    })}
                </div>
            </div>
        </div>
    )
}

// ── Small UI components ──

function UsageBar({ percent, color = 'blue', label, detail }) {
    const barColor = percent > 85 ? 'bg-red-500' : percent > 60 ? 'bg-yellow-500' : `bg-${color}-500`
    return (
        <div>
            <div className="flex justify-between text-xs mb-1">
                <span style={{ color: 'var(--apple-text-secondary)' }}>{label}</span>
                <span style={{ color: 'var(--apple-text-primary)' }}>{detail}</span>
            </div>
            <div className="w-full rounded-full h-2" style={{ background: 'var(--apple-bg-tertiary)' }}>
                <div className={`${barColor} h-2 rounded-full transition-all duration-500`} style={{ width: `${Math.min(percent, 100)}%` }} />
            </div>
        </div>
    )
}

function StatCard({ label, value, sub, color = 'gray' }) {
    const colorVar = {
        green: 'var(--apple-green)', yellow: 'var(--apple-yellow)', red: 'var(--apple-red)',
        blue: 'var(--apple-accent)', gray: 'var(--apple-text-primary)',
    }
    return (
        <div className="rounded-lg p-3" style={{ background: 'var(--apple-bg-tertiary)' }}>
            <div className="text-xs mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>{label}</div>
            <p className="text-lg font-bold" style={{ color: colorVar[color] || colorVar.gray }}>{value}</p>
            {sub && <p className="text-xs mt-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>{sub}</p>}
        </div>
    )
}

function TempBadge({ temp }) {
    if (temp == null) return <span style={{ color: 'var(--apple-text-quaternary)' }}>—</span>
    return <span className="font-mono" style={{ color: temp >= 85 ? 'var(--apple-red)' : temp >= 70 ? 'var(--apple-yellow)' : 'var(--apple-green)' }}>{temp}°C</span>
}

function StatusChip({ status, count }) {
    const chipStyle = {
        success: { background: 'var(--apple-green-bg)', color: 'var(--apple-green)', borderColor: 'var(--apple-green-border)' },
        deflected: { background: 'var(--apple-yellow-bg)', color: 'var(--apple-yellow)', borderColor: 'var(--apple-yellow-border)' },
        error: { background: 'var(--apple-red-bg)', color: 'var(--apple-red)', borderColor: 'var(--apple-red-border)' },
    }
    return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs border" style={chipStyle[status] || chipStyle.success}>
            {status} <span className="font-bold">{count}</span>
        </span>
    )
}

// ── Main component ──

export function ProductionTab() {
    const [stats, setStats] = useState(null)
    const [history, setHistory] = useState(null)
    const [metrics, setMetrics] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [timeRange, setTimeRange] = useState('1h')
    const intervalRef = useRef(null)

    // Client-side accumulated hardware history (from each poll)
    const [hwHistory, setHwHistory] = useState([])
    const [gpuHistory, setGpuHistory] = useState([])

    useEffect(() => {
        let cancelled = false
        async function loadData() {
            setLoading(true)
            try {
                const [statsData, historyData, metricsData] = await Promise.allSettled([
                    fetchStats(),
                    fetchBenchmarkHistory().catch(() => null),
                    fetchMetricsSummary().catch(() => null),
                ])
                if (cancelled) return
                if (statsData.status === 'fulfilled') setStats(statsData.value)
                if (historyData.status === 'fulfilled') setHistory(historyData.value)
                if (metricsData.status === 'fulfilled') {
                    const m = metricsData.value
                    setMetrics(m)
                    if (m?.hardware) appendHwSnapshot(m.hardware)
                }
            } catch (err) {
                if (!cancelled) setError(err.message)
            } finally {
                if (!cancelled) setLoading(false)
            }
        }

        function appendHwSnapshot(hw) {
            const t = new Date().toISOString()
            setHwHistory(prev => {
                const next = [...prev, { t, cpu: hw.cpu_percent, ram: hw.ram_percent }]
                return next.length > MAX_HW_HISTORY ? next.slice(-MAX_HW_HISTORY) : next
            })
            if (hw.gpus && hw.gpus.length > 0) {
                setGpuHistory(prev => {
                    const point = { t }
                    hw.gpus.forEach((g, i) => {
                        point[`gpu${i}_util`] = g.gpu_util_percent
                        point[`gpu${i}_temp`] = g.temperature_c
                        point[`gpu${i}_vram`] = g.vram_total_gb > 0
                            ? Math.round((g.vram_used_gb / g.vram_total_gb) * 100)
                            : 0
                    })
                    const next = [...prev, point]
                    return next.length > MAX_HW_HISTORY ? next.slice(-MAX_HW_HISTORY) : next
                })
            }
        }

        loadData()

        // Poll metrics every 10 seconds
        intervalRef.current = setInterval(async () => {
            try {
                const data = await fetchMetricsSummary()
                if (!cancelled) {
                    setMetrics(data)
                    if (data?.hardware) appendHwSnapshot(data.hardware)
                }
            } catch { /* silent */ }
        }, 10000)

        return () => {
            cancelled = true
            if (intervalRef.current) clearInterval(intervalRef.current)
        }
    }, [])

    if (loading) {
        return (
            <div className="flex-1 flex items-center justify-center">
                <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--apple-text-secondary)' }}>
                    <Spinner size={16} /> Loading production info...
                </div>
            </div>
        )
    }

    const lat = metrics?.latency || {}
    const hw = metrics?.hardware || {}
    const pipe = metrics?.pipeline || {}
    const qbs = metrics?.queries_by_status || {}
    const rangeMs = TIME_RANGES.find(r => r.key === timeRange)?.ms || 60 * 60 * 1000
    const lh = filterByTime(metrics?.latency_history || [], rangeMs)
    const filteredHw = filterByTime(hwHistory, rangeMs)
    const filteredGpu = filterByTime(gpuHistory, rangeMs)

    return (
        <div className="flex-1 overflow-y-auto">
            <div className="max-w-4xl mx-auto px-4 py-6">
                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'var(--apple-accent-bg)', border: '1px solid var(--apple-accent-border)' }}>
                        <CpuIcon size={20} />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold" style={{ color: 'var(--apple-text-primary)' }}>Production</h2>
                        <p className="text-xs" style={{ color: 'var(--apple-text-tertiary)' }}>Live metrics, hardware stats, and deployment info</p>
                    </div>
                    <div className="ml-auto flex items-center gap-3">
                        {/* Time range selector */}
                        <div className="inline-flex rounded-full p-0.5" style={{ background: 'var(--apple-bg-tertiary)' }}>
                            {TIME_RANGES.map(r => (
                                <button
                                    key={r.key}
                                    onClick={() => setTimeRange(r.key)}
                                    className="px-2.5 py-1 rounded-full text-[11px] font-medium transition-all"
                                    style={{
                                        background: timeRange === r.key ? 'var(--apple-accent)' : 'transparent',
                                        color: timeRange === r.key ? '#fff' : 'var(--apple-text-tertiary)',
                                    }}
                                >
                                    {r.label}
                                </button>
                            ))}
                        </div>
                        {metrics && (
                            <div className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--apple-text-tertiary)' }}>
                                <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: 'var(--apple-green)' }} />
                                Live
                            </div>
                        )}
                    </div>
                </div>

                {error && (
                    <div className="text-sm rounded-lg px-4 py-3 mb-6" style={{ color: 'var(--apple-red)', background: 'var(--apple-red-bg)', border: '1px solid var(--apple-red-border)' }}>
                        {error}
                    </div>
                )}

                {/* Traffic & Users */}
                {metrics && (
                    <div className="glass-card p-5 mb-6">
                        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                            <BarChartIcon size={16} /> Traffic & Users
                        </h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                            <StatCard label="Total Queries" value={metrics.total_queries.toLocaleString()} color="blue" />
                            <StatCard label="Unique Users" value={metrics.unique_users.toLocaleString()} color="green" />
                            <StatCard label="Uptime" value={formatUptime(metrics.uptime_seconds)} />
                            <StatCard
                                label="Success Rate"
                                value={metrics.total_queries > 0
                                    ? `${((qbs.success || 0) / metrics.total_queries * 100).toFixed(1)}%`
                                    : '—'}
                                color={metrics.total_queries > 0 && (qbs.success || 0) / metrics.total_queries >= 0.9 ? 'green' : 'yellow'}
                            />
                        </div>
                        <div className="flex items-center gap-2 flex-wrap">
                            {Object.entries(qbs).map(([status, count]) => (
                                <StatusChip key={status} status={status} count={count} />
                            ))}
                        </div>
                        <TimeChart
                            data={lh}
                            rangeMs={rangeMs}
                            title="Queries Over Time"
                            emptyMessage="No queries yet"
                            yFormat={v => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${Math.round(v)}`}
                            series={[
                                { key: 'n', color: '#38bdf8', label: 'Total Queries', fill: true },
                            ]}
                        />
                    </div>
                )}

                {/* Latency */}
                {metrics && (
                    <div className="glass-card p-5 mb-6">
                        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                            <ZapIcon size={16} /> Latency
                        </h3>
                        {metrics.total_queries > 0 ? (
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                <StatCard label="p50" value={`${(lat.p50_ms / 1000).toFixed(1)}s`}
                                    color={lat.p50_ms < 3000 ? 'green' : lat.p50_ms < 8000 ? 'yellow' : 'red'} />
                                <StatCard label="p90" value={`${(lat.p90_ms / 1000).toFixed(1)}s`}
                                    color={lat.p90_ms < 5000 ? 'green' : lat.p90_ms < 12000 ? 'yellow' : 'red'} />
                                <StatCard label="p99" value={`${(lat.p99_ms / 1000).toFixed(1)}s`}
                                    color={lat.p99_ms < 10000 ? 'green' : lat.p99_ms < 20000 ? 'yellow' : 'red'} />
                                <StatCard label="Average" value={`${(lat.avg_ms / 1000).toFixed(1)}s`} />
                            </div>
                        ) : (
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                <StatCard label="p50" value="—" />
                                <StatCard label="p90" value="—" />
                                <StatCard label="p99" value="—" />
                                <StatCard label="Average" value="—" />
                            </div>
                        )}
                        <TimeChart
                            data={lh}
                            rangeMs={rangeMs}
                            title="Latency Percentiles Over Time"
                            emptyMessage="Waiting for queries — ask a question to see latency data"
                            series={[
                                { key: 'p99', color: '#f87171', label: 'p99', fill: true },
                                { key: 'p90', color: '#facc15', label: 'p90' },
                                { key: 'p50', color: '#4ade80', label: 'p50' },
                            ]}
                        />
                    </div>
                )}

                {/* Pipeline Breakdown Over Time */}
                {metrics && (
                    <div className="glass-card p-5 mb-6">
                        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                            <DatabaseIcon size={16} /> Pipeline Breakdown
                        </h3>
                        {/* Avg breakdown bar */}
                        {(pipe.avg_retrieval_ms > 0 || pipe.avg_generation_ms > 0) && (
                            <div className="mb-2">
                                <div className="text-xs mb-2" style={{ color: 'var(--apple-text-tertiary)' }}>Average Breakdown</div>
                                <div className="flex h-4 rounded-full overflow-hidden" style={{ background: 'var(--apple-bg-tertiary)' }}>
                                    {pipe.avg_retrieval_ms > 0 && (
                                        <div className="bg-blue-500 flex items-center justify-center text-[9px] font-mono text-white"
                                            style={{ width: `${Math.max((pipe.avg_retrieval_ms / lat.avg_ms) * 100, 10)}%` }}
                                            title={`Retrieval: ${(pipe.avg_retrieval_ms / 1000).toFixed(2)}s`}>Retrieval</div>
                                    )}
                                    {pipe.avg_reranking_ms > 0 && (
                                        <div className="bg-purple-500 flex items-center justify-center text-[9px] font-mono text-white"
                                            style={{ width: `${Math.max((pipe.avg_reranking_ms / lat.avg_ms) * 100, 10)}%` }}
                                            title={`Reranking: ${(pipe.avg_reranking_ms / 1000).toFixed(2)}s`}>Rerank</div>
                                    )}
                                    {pipe.avg_generation_ms > 0 && (
                                        <div className="bg-emerald-500 flex items-center justify-center text-[9px] font-mono text-white"
                                            style={{ width: `${Math.max((pipe.avg_generation_ms / lat.avg_ms) * 100, 10)}%` }}
                                            title={`Generation: ${(pipe.avg_generation_ms / 1000).toFixed(2)}s`}>Generation</div>
                                    )}
                                </div>
                                <div className="flex gap-4 mt-1.5 text-[10px]" style={{ color: 'var(--apple-text-tertiary)' }}>
                                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500" />Retrieval {(pipe.avg_retrieval_ms / 1000).toFixed(2)}s</span>
                                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500" />Rerank {(pipe.avg_reranking_ms / 1000).toFixed(2)}s</span>
                                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500" />Generation {(pipe.avg_generation_ms / 1000).toFixed(2)}s</span>
                                </div>
                            </div>
                        )}
                        <TimeChart
                            data={lh}
                            rangeMs={rangeMs}
                            title="Per-Query Pipeline Timing"
                            emptyMessage="Waiting for queries — pipeline timing will appear here"
                            series={[
                                { key: 'generation_ms', color: '#34d399', label: 'Generation', fill: true },
                                { key: 'retrieval_ms', color: '#60a5fa', label: 'Retrieval', fill: true },
                                { key: 'reranking_ms', color: '#a78bfa', label: 'Reranking' },
                            ]}
                        />
                    </div>
                )}

                {/* Hardware */}
                {metrics && hw.cpu_percent != null && (
                    <div className="glass-card p-5 mb-6">
                        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                            <CpuIcon size={16} /> Hardware
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                            <UsageBar percent={hw.cpu_percent} color="blue" label="CPU" detail={`${hw.cpu_percent.toFixed(1)}%`} />
                            <UsageBar percent={hw.ram_percent} color="purple" label="RAM" detail={`${hw.ram_used_gb} / ${hw.ram_total_gb} GB`} />
                        </div>

                        {/* CPU & RAM over time */}
                        <TimeChart
                            data={filteredHw}
                            rangeMs={rangeMs}
                            title="CPU & RAM Over Time"
                            emptyMessage="Collecting data — chart will appear after two polling intervals"
                            yFormat={v => `${Math.round(v)}%`}
                            series={[
                                { key: 'cpu', color: '#60a5fa', label: 'CPU %', fill: true },
                                { key: 'ram', color: '#c084fc', label: 'RAM %' },
                            ]}
                        />

                        {hw.gpus && hw.gpus.length > 0 && (
                            <div className="space-y-3 mt-4">
                                {hw.gpus.map((gpu, i) => (
                                    <div key={i} className="rounded-xl p-4" style={{ background: 'var(--apple-bg-tertiary)' }}>
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="text-xs font-medium" style={{ color: 'var(--apple-text-primary)' }}>{gpu.name}</span>
                                            <TempBadge temp={gpu.temperature_c} />
                                        </div>
                                        <div className="grid grid-cols-2 gap-3">
                                            <UsageBar percent={gpu.gpu_util_percent} color="green" label="GPU Utilization" detail={`${gpu.gpu_util_percent}%`} />
                                            <UsageBar percent={(gpu.vram_used_gb / gpu.vram_total_gb) * 100} color="yellow" label="VRAM" detail={`${gpu.vram_used_gb} / ${gpu.vram_total_gb} GB`} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* GPU utilization & temp over time */}
                        {hw.gpus && hw.gpus.map((gpu, i) => (
                            <TimeChart
                                key={`gpu-chart-${i}`}
                                data={filteredGpu}
                                rangeMs={rangeMs}
                                title={`${gpu.name} — Utilization & Temperature`}
                                emptyMessage="Collecting data — GPU chart will appear after two polling intervals"
                                yFormat={v => `${Math.round(v)}`}
                                series={[
                                    { key: `gpu${i}_util`, color: '#4ade80', label: 'GPU %', fill: true },
                                    { key: `gpu${i}_vram`, color: '#facc15', label: 'VRAM %' },
                                    { key: `gpu${i}_temp`, color: '#f87171', label: 'Temp °C' },
                                ]}
                            />
                        ))}
                    </div>
                )}

                {/* Pipeline Configuration */}
                {stats && (
                    <div className="glass-card p-5 mb-6">
                        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                            <DatabaseIcon size={16} /> Pipeline Configuration
                        </h3>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            {[
                                { label: 'Papers', value: stats.total_papers, icon: BookIcon },
                                { label: 'Chunks', value: stats.total_chunks?.toLocaleString(), icon: DatabaseIcon },
                                { label: 'Embedding Model', value: (stats.embedding_model || '').split('/')[1] || stats.embedding_model, icon: CpuIcon },
                                { label: 'LLM Backend', value: stats.llm_backend, icon: ZapIcon },
                                { label: 'LLM Model', value: stats.llm_model, icon: CpuIcon },
                                { label: 'Collection', value: stats.collection_name, icon: SearchIcon },
                            ].map(({ label, value, icon: Ic }) => (
                                <div key={label} className="rounded-xl p-4" style={{ background: 'var(--apple-bg-tertiary)' }}>
                                    <div className="flex items-center gap-1.5 text-xs mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>
                                        <Ic size={12} /> {label}
                                    </div>
                                    <p className="text-sm font-medium" style={{ color: 'var(--apple-text-primary)' }}>{value}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* System Health */}
                <div className="glass-card p-5 mb-6">
                    <h3 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                        <ZapIcon size={16} /> System Health
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="rounded-xl p-4" style={{ background: 'var(--apple-bg-tertiary)' }}>
                            <div className="text-xs mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>API Status</div>
                            <div className="flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--apple-green)' }}></span>
                                <span className="text-sm" style={{ color: 'var(--apple-green)' }}>Online</span>
                            </div>
                        </div>
                        <div className="rounded-xl p-4" style={{ background: 'var(--apple-bg-tertiary)' }}>
                            <div className="text-xs mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>Pipeline Ready</div>
                            <div className="flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full" style={{ background: stats ? 'var(--apple-green)' : 'var(--apple-yellow)' }}></span>
                                <span className="text-sm" style={{ color: stats ? 'var(--apple-green)' : 'var(--apple-yellow)' }}>
                                    {stats ? 'Ready' : 'Loading'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Recent Queries */}
                {metrics && metrics.recent_queries && metrics.recent_queries.length > 0 && (
                    <div className="glass-card p-5 mb-6">
                        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                            <SearchIcon size={16} /> Recent Queries
                        </h3>
                        <div className="space-y-1.5 max-h-64 overflow-y-auto">
                            {metrics.recent_queries.slice(0, 20).map((q, i) => (
                                <div key={i} className="flex items-center gap-3 text-xs rounded-xl p-2.5" style={{ background: 'var(--apple-bg-tertiary)' }}>
                                    <span className="font-mono shrink-0 w-16" style={{ color: 'var(--apple-text-quaternary)' }}>
                                        {new Date(q.timestamp).toLocaleTimeString()}
                                    </span>
                                    <span className="flex-1 truncate" style={{ color: 'var(--apple-text-primary)' }} title={q.question}>
                                        {q.question}
                                    </span>
                                    <span className="font-mono shrink-0" style={{ color: q.latency_ms < 3000 ? 'var(--apple-green)' : q.latency_ms < 8000 ? 'var(--apple-yellow)' : 'var(--apple-red)' }}>
                                        {(q.latency_ms / 1000).toFixed(1)}s
                                    </span>
                                    <span className="shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium border" style={
                                        q.status === 'success' ? { background: 'var(--apple-green-bg)', color: 'var(--apple-green)', borderColor: 'var(--apple-green-border)' } :
                                        q.status === 'deflected' ? { background: 'var(--apple-yellow-bg)', color: 'var(--apple-yellow)', borderColor: 'var(--apple-yellow-border)' } :
                                        { background: 'var(--apple-red-bg)', color: 'var(--apple-red)', borderColor: 'var(--apple-red-border)' }
                                    }>
                                        {q.status}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Evaluation History */}
                <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                        <BarChartIcon size={16} /> Evaluation History
                    </h3>
                    {history && history.runs && history.runs.length > 0 ? (
                        <div className="space-y-2">
                            {history.runs.map((run, i) => (
                                <div key={i} className="flex items-center gap-3 text-xs rounded-xl p-2.5" style={{ background: 'var(--apple-bg-tertiary)' }}>
                                    <span className="font-mono" style={{ color: 'var(--apple-text-quaternary)' }}>{run.timestamp}</span>
                                    <span style={{ color: 'var(--apple-accent)' }}>{run.benchmark}</span>
                                    <span className="flex-1" style={{ color: 'var(--apple-text-secondary)' }}>{run.total_evaluated} entries</span>
                                    {run.accuracy !== undefined && (
                                        <span className="font-bold" style={{ color: run.accuracy >= 0.8 ? 'var(--apple-green)' : run.accuracy >= 0.6 ? 'var(--apple-yellow)' : 'var(--apple-red)' }}>
                                            {(run.accuracy * 100).toFixed(1)}%
                                        </span>
                                    )}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-xs" style={{ color: 'var(--apple-text-quaternary)' }}>No evaluation runs recorded yet. Run a benchmark from the Benchmarks tab to see history here.</p>
                    )}
                </div>
            </div>
        </div>
    )
}
