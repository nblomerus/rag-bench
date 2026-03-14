import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { fetchGraphContext } from '../utils/api'
import { ChevronDown, ChevronUp } from './Icons'

// Entity type → color mapping (matches project entity types)
const TYPE_COLORS = {
    MODEL:   { fill: 'rgba(10,132,255,0.15)',  stroke: 'rgba(10,132,255,0.6)',  text: 'rgb(10,132,255)' },
    DATASET: { fill: 'rgba(48,209,88,0.15)',   stroke: 'rgba(48,209,88,0.6)',   text: 'rgb(48,209,88)' },
    METHOD:  { fill: 'rgba(175,82,222,0.15)',   stroke: 'rgba(175,82,222,0.6)',  text: 'rgb(175,82,222)' },
    METRIC:  { fill: 'rgba(255,159,10,0.15)',   stroke: 'rgba(255,159,10,0.6)',  text: 'rgb(255,159,10)' },
    TASK:    { fill: 'rgba(255,69,58,0.15)',    stroke: 'rgba(255,69,58,0.6)',   text: 'rgb(255,69,58)' },
    TOOL:    { fill: 'rgba(100,210,255,0.15)',  stroke: 'rgba(100,210,255,0.6)', text: 'rgb(100,210,255)' },
}

const DEFAULT_COLOR = { fill: 'rgba(142,142,147,0.15)', stroke: 'rgba(142,142,147,0.5)', text: 'rgb(142,142,147)' }

function getColor(entityType) {
    return TYPE_COLORS[entityType] || DEFAULT_COLOR
}

// Truncate long names for display
function truncName(name, max = 18) {
    return name.length > max ? name.slice(0, max - 1) + '…' : name
}

// Simple force-directed layout simulation
function runSimulation(nodes, edges, width, height, iterations = 120) {
    const k = Math.sqrt((width * height) / Math.max(nodes.length, 1)) * 0.6
    const positions = {}

    // Initialize positions in a circle around center
    const cx = width / 2, cy = height / 2
    const radius = Math.min(width, height) * 0.3
    nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / nodes.length
        positions[node.id] = {
            x: cx + radius * Math.cos(angle) + (Math.random() - 0.5) * 20,
            y: cy + radius * Math.sin(angle) + (Math.random() - 0.5) * 20,
            vx: 0, vy: 0,
        }
    })

    // Build edge lookup for connected nodes
    const edgeSet = new Set()
    edges.forEach(e => {
        edgeSet.add(`${e.source}|${e.target}`)
        edgeSet.add(`${e.target}|${e.source}`)
    })

    for (let iter = 0; iter < iterations; iter++) {
        const temp = 1 - iter / iterations // cooling
        const force = {}
        nodes.forEach(n => { force[n.id] = { fx: 0, fy: 0 } })

        // Repulsion between all node pairs
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = positions[nodes[i].id]
                const b = positions[nodes[j].id]
                let dx = a.x - b.x, dy = a.y - b.y
                const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
                const repForce = (k * k) / dist
                const fx = (dx / dist) * repForce
                const fy = (dy / dist) * repForce
                force[nodes[i].id].fx += fx
                force[nodes[i].id].fy += fy
                force[nodes[j].id].fx -= fx
                force[nodes[j].id].fy -= fy
            }
        }

        // Attraction along edges
        edges.forEach(e => {
            const a = positions[e.source]
            const b = positions[e.target]
            if (!a || !b) return
            let dx = b.x - a.x, dy = b.y - a.y
            const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
            const attForce = (dist * dist) / k * 0.3
            const fx = (dx / dist) * attForce
            const fy = (dy / dist) * attForce
            force[e.source].fx += fx
            force[e.source].fy += fy
            force[e.target].fx -= fx
            force[e.target].fy -= fy
        })

        // Center gravity
        nodes.forEach(n => {
            const p = positions[n.id]
            force[n.id].fx += (cx - p.x) * 0.01
            force[n.id].fy += (cy - p.y) * 0.01
        })

        // Apply forces with cooling
        const maxDisp = Math.max(width, height) * 0.1 * temp
        nodes.forEach(n => {
            const f = force[n.id]
            const p = positions[n.id]
            const disp = Math.sqrt(f.fx * f.fx + f.fy * f.fy)
            if (disp > 0) {
                const capped = Math.min(disp, maxDisp)
                p.x += (f.fx / disp) * capped
                p.y += (f.fy / disp) * capped
            }
            // Keep in bounds with padding
            const pad = 40
            p.x = Math.max(pad, Math.min(width - pad, p.x))
            p.y = Math.max(pad, Math.min(height - pad, p.y))
        })
    }

    return positions
}


/**
 * KnowledgeGraph — Interactive force-directed knowledge graph visualization.
 *
 * Props:
 * - question: string — the query to match entities for
 */
export function KnowledgeGraph({ question }) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(false)
    const [expanded, setExpanded] = useState(false)
    const [hoveredNode, setHoveredNode] = useState(null)
    const [hoveredEdge, setHoveredEdge] = useState(null)
    const svgRef = useRef(null)

    // Fetch graph context when expanded
    useEffect(() => {
        if (!expanded || !question || data) return
        let cancelled = false
        setLoading(true)
        fetchGraphContext(question).then(result => {
            if (!cancelled) {
                setData(result?.nodes?.length > 0 ? result : null)
                setLoading(false)
            }
        }).catch(() => {
            if (!cancelled) setLoading(false)
        })
        return () => { cancelled = true }
    }, [expanded, question])

    // Compute layout
    const WIDTH = 560
    const HEIGHT = 340
    const layout = useMemo(() => {
        if (!data?.nodes?.length) return null
        return runSimulation(data.nodes, data.edges, WIDTH, HEIGHT)
    }, [data])

    const handleToggle = useCallback(() => {
        setExpanded(v => !v)
    }, [])

    // Don't render anything until expanded at least once, then show even if empty
    if (!expanded && !data) {
        return (
            <button
                onClick={handleToggle}
                className="mt-1.5 w-full flex items-center gap-2 px-3 py-2 rounded-xl text-left transition-colors"
                style={{
                    background: 'var(--apple-glass-bg)',
                    border: '1px solid var(--apple-glass-border)',
                    backdropFilter: 'blur(12px)',
                }}
                onMouseOver={e => e.currentTarget.style.background = 'var(--apple-bg-tertiary)'}
                onMouseOut={e => e.currentTarget.style.background = 'var(--apple-glass-bg)'}
            >
                <GraphIcon size={12} />
                <span className="text-[11px] font-medium" style={{ color: 'var(--apple-text-secondary)' }}>
                    Knowledge Graph
                </span>
                <span className="text-[10px] ml-auto" style={{ color: 'var(--apple-text-quaternary)' }}>
                    Explore entities
                </span>
                <ChevronDown size={10} />
            </button>
        )
    }

    return (
        <div className="mt-1.5">
            <button
                onClick={handleToggle}
                className="w-full flex items-center gap-2 px-3 py-2 rounded-xl text-left transition-colors"
                style={{
                    background: 'var(--apple-glass-bg)',
                    border: '1px solid var(--apple-glass-border)',
                    backdropFilter: 'blur(12px)',
                }}
                onMouseOver={e => e.currentTarget.style.background = 'var(--apple-bg-tertiary)'}
                onMouseOut={e => e.currentTarget.style.background = 'var(--apple-glass-bg)'}
            >
                <GraphIcon size={12} />
                <span className="text-[11px] font-medium" style={{ color: 'var(--apple-text-secondary)' }}>
                    Knowledge Graph
                </span>
                {data && (
                    <span className="text-[10px]" style={{ color: 'var(--apple-text-quaternary)' }}>
                        {data.nodes.length} entities · {data.edges.length} relations
                    </span>
                )}
                <span className="ml-auto">
                    {expanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
                </span>
            </button>

            {expanded && (
                <div className="mt-1 rounded-xl overflow-hidden"
                    style={{
                        background: 'var(--apple-glass-bg)',
                        border: '1px solid var(--apple-glass-border)',
                        backdropFilter: 'blur(12px)',
                    }}>
                    {loading && (
                        <div className="flex items-center justify-center py-8">
                            <span className="text-[11px]" style={{ color: 'var(--apple-text-quaternary)' }}>
                                Loading graph...
                            </span>
                        </div>
                    )}

                    {!loading && !data && (
                        <div className="flex items-center justify-center py-6">
                            <span className="text-[11px]" style={{ color: 'var(--apple-text-quaternary)' }}>
                                No graph entities found for this query
                            </span>
                        </div>
                    )}

                    {!loading && data && layout && (
                        <>
                            <svg
                                ref={svgRef}
                                viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
                                className="w-full"
                                style={{ height: '340px', cursor: 'default' }}
                            >
                                <defs>
                                    <marker id="arrow" viewBox="0 0 10 6" refX="10" refY="3"
                                        markerWidth="8" markerHeight="6" orient="auto-start-reverse">
                                        <path d="M 0 0 L 10 3 L 0 6 z" fill="var(--apple-text-quaternary)" opacity="0.5" />
                                    </marker>
                                </defs>

                                {/* Edges */}
                                {data.edges.map((edge, i) => {
                                    const from = layout[edge.source]
                                    const to = layout[edge.target]
                                    if (!from || !to) return null

                                    const isHovered = hoveredEdge === i
                                    const isConnectedToHoveredNode = hoveredNode &&
                                        (edge.source === hoveredNode || edge.target === hoveredNode)
                                    const dimmed = hoveredNode && !isConnectedToHoveredNode

                                    // Offset endpoint slightly so arrow doesn't overlap node
                                    const dx = to.x - from.x, dy = to.y - from.y
                                    const dist = Math.sqrt(dx * dx + dy * dy)
                                    const nodeR = 22
                                    const sx = from.x + (dx / dist) * nodeR
                                    const sy = from.y + (dy / dist) * nodeR
                                    const ex = to.x - (dx / dist) * nodeR
                                    const ey = to.y - (dy / dist) * nodeR
                                    // Midpoint for label
                                    const mx = (sx + ex) / 2, my = (sy + ey) / 2

                                    return (
                                        <g key={`edge-${i}`}
                                            onMouseEnter={() => setHoveredEdge(i)}
                                            onMouseLeave={() => setHoveredEdge(null)}
                                        >
                                            <line
                                                x1={sx} y1={sy} x2={ex} y2={ey}
                                                stroke={isHovered || isConnectedToHoveredNode ? 'var(--apple-accent)' : 'var(--apple-text-quaternary)'}
                                                strokeWidth={isHovered ? 2 : Math.min(1 + edge.weight * 0.3, 3)}
                                                opacity={dimmed ? 0.15 : (isHovered || isConnectedToHoveredNode ? 0.8 : 0.35)}
                                                markerEnd="url(#arrow)"
                                            />
                                            {/* Wider invisible hitbox for easier hovering */}
                                            <line
                                                x1={sx} y1={sy} x2={ex} y2={ey}
                                                stroke="transparent" strokeWidth="12"
                                            />
                                            {/* Edge label on hover */}
                                            {(isHovered || isConnectedToHoveredNode) && (
                                                <g transform={`translate(${mx}, ${my})`}>
                                                    <rect
                                                        x={-edge.predicate.length * 3 - 6} y={-8}
                                                        width={edge.predicate.length * 6 + 12} height={16}
                                                        rx={4}
                                                        fill="var(--apple-bg-primary)" stroke="var(--apple-divider)"
                                                        strokeWidth="0.5" opacity="0.95"
                                                    />
                                                    <text
                                                        textAnchor="middle" dominantBaseline="central"
                                                        fontSize="8" fontWeight="500" fontFamily="system-ui"
                                                        fill="var(--apple-text-secondary)"
                                                    >
                                                        {edge.predicate.replace(/_/g, ' ')}
                                                    </text>
                                                </g>
                                            )}
                                        </g>
                                    )
                                })}

                                {/* Nodes */}
                                {data.nodes.map(node => {
                                    const pos = layout[node.id]
                                    if (!pos) return null
                                    const color = getColor(node.entity_type)
                                    const isMatched = node.matched
                                    const isHovered = hoveredNode === node.id
                                    const isConnected = hoveredNode && data.edges.some(
                                        e => (e.source === hoveredNode && e.target === node.id) ||
                                             (e.target === hoveredNode && e.source === node.id)
                                    )
                                    const dimmed = hoveredNode && hoveredNode !== node.id && !isConnected
                                    const r = isMatched ? 24 : 20

                                    return (
                                        <g key={node.id}
                                            transform={`translate(${pos.x}, ${pos.y})`}
                                            onMouseEnter={() => setHoveredNode(node.id)}
                                            onMouseLeave={() => setHoveredNode(null)}
                                            style={{ cursor: 'pointer' }}
                                            opacity={dimmed ? 0.25 : 1}
                                        >
                                            {/* Glow for matched entities */}
                                            {isMatched && (
                                                <circle r={r + 4} fill={color.stroke} opacity={0.15} />
                                            )}

                                            {/* Node circle */}
                                            <circle
                                                r={r}
                                                fill={color.fill}
                                                stroke={isHovered ? color.text : color.stroke}
                                                strokeWidth={isMatched ? 2 : 1.5}
                                            />

                                            {/* Entity name */}
                                            <text
                                                textAnchor="middle" dominantBaseline="central"
                                                fontSize={isMatched ? '8.5' : '7.5'}
                                                fontWeight={isMatched ? '600' : '500'}
                                                fontFamily="system-ui"
                                                fill={color.text}
                                            >
                                                {truncName(node.name, isMatched ? 16 : 14)}
                                            </text>

                                            {/* Type badge (small, below) */}
                                            <text
                                                textAnchor="middle" y={r + 10}
                                                fontSize="6" fontFamily="system-ui"
                                                fill="var(--apple-text-quaternary)"
                                            >
                                                {node.entity_type}
                                            </text>

                                            {/* Tooltip on hover */}
                                            {isHovered && (
                                                <g transform={`translate(0, ${-r - 16})`}>
                                                    <rect
                                                        x={-node.name.length * 3.2 - 8} y={-10}
                                                        width={node.name.length * 6.4 + 16} height={20}
                                                        rx={6}
                                                        fill="var(--apple-bg-primary)"
                                                        stroke="var(--apple-divider)" strokeWidth="0.5"
                                                        filter="drop-shadow(0 2px 4px rgba(0,0,0,0.15))"
                                                    />
                                                    <text
                                                        textAnchor="middle" dominantBaseline="central"
                                                        fontSize="9" fontWeight="600" fontFamily="system-ui"
                                                        fill="var(--apple-text-primary)"
                                                    >
                                                        {node.name}
                                                    </text>
                                                </g>
                                            )}
                                        </g>
                                    )
                                })}
                            </svg>

                            {/* Legend */}
                            <div className="px-3 py-2 flex flex-wrap gap-3 items-center" style={{ borderTop: '1px solid var(--apple-divider)' }}>
                                {Object.entries(TYPE_COLORS).map(([type, color]) => {
                                    const hasType = data.nodes.some(n => n.entity_type === type)
                                    if (!hasType) return null
                                    return (
                                        <span key={type} className="flex items-center gap-1.5">
                                            <span className="w-2 h-2 rounded-full" style={{ background: color.text }} />
                                            <span className="text-[9px]" style={{ color: 'var(--apple-text-tertiary)' }}>{type}</span>
                                        </span>
                                    )
                                })}
                                <span className="text-[9px] ml-auto" style={{ color: 'var(--apple-text-quaternary)' }}>
                                    Matched: {data.matched_entities?.join(', ')}
                                </span>
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    )
}


// Graph/network icon
function GraphIcon({ size = 16 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="6" cy="6" r="2" />
            <circle cx="18" cy="6" r="2" />
            <circle cx="6" cy="18" r="2" />
            <circle cx="18" cy="18" r="2" />
            <circle cx="12" cy="12" r="2" />
            <line x1="7.8" y1="7.2" x2="10.5" y2="10.5" />
            <line x1="13.5" y1="10.5" x2="16.2" y2="7.2" />
            <line x1="7.8" y1="16.8" x2="10.5" y2="13.5" />
            <line x1="13.5" y1="13.5" x2="16.2" y2="16.8" />
        </svg>
    )
}
