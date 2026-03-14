import React, { useState } from 'react'
import { LayersIcon, SearchIcon, ActivityIcon, ShieldIcon, FilterIcon, CpuIcon, CheckCircleIcon, ChevronDown, ChevronUp, GitBranchIcon } from './Icons'

const STAGE_CONFIG = {
    classify: { icon: GitBranchIcon, label: 'Query Classification' },
    retrieval: { icon: SearchIcon, label: 'Hybrid Retrieval' },
    reranking: { icon: ActivityIcon, label: 'Cross-Encoder Reranking' },
    crag: { icon: ShieldIcon, label: 'CRAG Confidence' },
    refinement: { icon: FilterIcon, label: 'Knowledge Refinement' },
    generation: { icon: CpuIcon, label: 'Answer Generation' },
}

const CONFIDENCE_STYLES = {
    correct: { color: 'var(--apple-green)', bg: 'var(--apple-green-bg)', border: 'var(--apple-green-border)', label: 'CORRECT' },
    ambiguous: { color: 'var(--apple-yellow)', bg: 'var(--apple-yellow-bg)', border: 'var(--apple-yellow-border)', label: 'AMBIGUOUS' },
    incorrect: { color: 'var(--apple-red)', bg: 'var(--apple-red-bg)', border: 'var(--apple-red-border)', label: 'INCORRECT' },
    unknown: { color: 'var(--apple-text-quaternary)', bg: 'var(--apple-bg-tertiary)', border: 'var(--apple-divider)', label: 'N/A' },
}

const STATUS_COLORS = {
    done: 'var(--apple-green)',
    running: 'var(--apple-accent)',
    skipped: 'var(--apple-text-quaternary)',
    correct: 'var(--apple-green)',
    ambiguous: 'var(--apple-yellow)',
    incorrect: 'var(--apple-red)',
}

function LiveStageRow({ stage, isLast }) {
    const config = STAGE_CONFIG[stage.stage] || { icon: CheckCircleIcon, label: stage.stage }
    const Icon = config.icon
    const isCrag = stage.stage === 'crag'
    const isRunning = stage.status === 'running'
    const conf = isCrag ? CONFIDENCE_STYLES[stage.status] || CONFIDENCE_STYLES.unknown : null
    const dotColor = isCrag && conf ? conf.color : (STATUS_COLORS[stage.status] || 'var(--apple-text-quaternary)')
    const dotBg = isCrag && conf ? conf.bg : (isRunning ? 'var(--apple-accent-bg, rgba(10,132,255,0.1))' : 'var(--apple-bg-tertiary)')
    const dotBorder = isCrag && conf ? conf.border : (isRunning ? 'var(--apple-accent)' : 'var(--apple-divider)')

    return (
        <div className="flex items-start gap-2.5" style={{ minHeight: '28px' }}>
            {/* Timeline dot + line */}
            <div className="flex flex-col items-center flex-shrink-0" style={{ width: '18px' }}>
                <div className={`w-[18px] h-[18px] rounded-full flex items-center justify-center flex-shrink-0 ${isRunning ? 'animate-pulse' : ''}`}
                    style={{
                        background: dotBg,
                        border: `1.5px solid ${dotBorder}`,
                        color: dotColor,
                    }}>
                    <Icon size={9} />
                </div>
                {!isLast && (
                    <div style={{ width: '1.5px', height: '10px', background: 'var(--apple-divider)' }} />
                )}
            </div>

            {/* Content */}
            <div className="flex-1" style={{ marginTop: '0px' }}>
                <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[11px] font-medium" style={{ color: dotColor }}>
                        {config.label}
                    </span>
                    {isCrag && conf && stage.status !== 'running' && (
                        <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded-full"
                            style={{ background: conf.bg, color: conf.color, border: `1px solid ${conf.border}` }}>
                            {conf.label}
                        </span>
                    )}
                    {stage.data?.duration_ms > 0 && (
                        <span className="text-[10px]" style={{ color: 'var(--apple-text-quaternary)' }}>
                            {stage.data.duration_ms.toFixed(0)}ms
                        </span>
                    )}
                    {stage.status === 'skipped' && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded-full"
                            style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-quaternary)', border: '1px solid var(--apple-divider)' }}>
                            SKIPPED
                        </span>
                    )}
                    {stage.data?.query_type && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
                            style={{
                                background: stage.data.query_type === 'multi_hop' ? 'rgba(175,82,222,0.12)' : 'var(--apple-bg-tertiary)',
                                color: stage.data.query_type === 'multi_hop' ? 'rgb(175,82,222)' : 'var(--apple-text-secondary)',
                                border: `1px solid ${stage.data.query_type === 'multi_hop' ? 'rgba(175,82,222,0.25)' : 'var(--apple-divider)'}`,
                            }}>
                            {stage.data.query_type === 'multi_hop' ? 'Multi-hop' : stage.data.query_type === 'entity' ? 'Entity-heavy' : 'Simple'}
                        </span>
                    )}
                    {stage.data?.sources_by_type && Object.keys(stage.data.sources_by_type).length > 0 && (
                        Object.entries(stage.data.sources_by_type).map(([type, count]) => (
                            <span key={type} className="text-[9px] px-1.5 py-0.5 rounded-full"
                                style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-tertiary)', border: '1px solid var(--apple-divider)' }}>
                                {type}: {count}
                            </span>
                        ))
                    )}
                </div>
                <p className="text-[10px] mt-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>
                    {stage.detail}
                </p>
            </div>
        </div>
    )
}

/**
 * PipelineInsight — shows the RAG pipeline decision trace.
 *
 * Two modes:
 * 1. Live mode (streaming): Shows stages as they arrive in real-time
 * 2. Final mode (after done): Shows the complete pipeline summary
 *
 * Props:
 * - stages: Array of live pipeline stage events (streaming mode)
 * - pipeline: Final pipeline summary object (done mode)
 * - live: Whether we're in streaming mode
 */
export function PipelineInsight({ stages, pipeline, live }) {
    const [expanded, setExpanded] = useState(true) // default open so users see it

    // Live mode: show stages as they stream in
    if (live && stages?.length > 0) {
        return (
            <div className="mt-1.5 px-3.5 py-2.5 rounded-xl"
                style={{
                    background: 'var(--apple-glass-bg)',
                    border: '1px solid var(--apple-glass-border)',
                    backdropFilter: 'blur(12px)',
                }}>
                <div className="flex items-center gap-2 mb-2">
                    <LayersIcon size={11} />
                    <span className="text-[11px] font-semibold" style={{ color: 'var(--apple-text-primary)' }}>
                        Pipeline
                    </span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded-full animate-pulse"
                        style={{ background: 'rgba(10,132,255,0.12)', color: 'var(--apple-accent)', border: '1px solid rgba(10,132,255,0.25)' }}>
                        LIVE
                    </span>
                </div>
                <div>
                    {stages.map((stage, i) => (
                        <LiveStageRow key={stage.stage} stage={stage} isLast={i === stages.length - 1} />
                    ))}
                </div>
            </div>
        )
    }

    // Final mode: show completed pipeline summary
    if (!pipeline) return null

    const cragConf = CONFIDENCE_STYLES[pipeline.crag_confidence] || CONFIDENCE_STYLES.unknown
    const queryTypeLabel = pipeline.query_type === 'multi_hop' ? 'Multi-hop' : pipeline.query_type === 'entity' ? 'Entity-heavy' : 'Simple'
    const pipelineStages = pipeline.stages || []

    return (
        <div className="mt-1.5">
            {/* Clickable summary bar — always visible */}
            <button
                onClick={() => setExpanded(v => !v)}
                className="w-full flex items-center gap-2 px-3 py-2 rounded-xl text-left transition-colors"
                style={{
                    background: 'var(--apple-glass-bg)',
                    border: '1px solid var(--apple-glass-border)',
                    backdropFilter: 'blur(12px)',
                }}
                onMouseOver={e => e.currentTarget.style.background = 'var(--apple-bg-tertiary)'}
                onMouseOut={e => e.currentTarget.style.background = 'var(--apple-glass-bg)'}
            >
                <LayersIcon size={12} />
                <span className="text-[11px] font-medium" style={{ color: 'var(--apple-text-secondary)' }}>
                    Pipeline
                </span>

                {/* Query type */}
                <span className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
                    style={{
                        background: pipeline.query_type === 'multi_hop' ? 'rgba(175,82,222,0.12)' : 'var(--apple-bg-tertiary)',
                        color: pipeline.query_type === 'multi_hop' ? 'rgb(175,82,222)' : 'var(--apple-text-secondary)',
                        border: `1px solid ${pipeline.query_type === 'multi_hop' ? 'rgba(175,82,222,0.25)' : 'var(--apple-divider)'}`,
                    }}>
                    {queryTypeLabel}
                </span>

                {/* CRAG badge */}
                {pipeline.crag_confidence && (
                    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded-full"
                        style={{ background: cragConf.bg, color: cragConf.color, border: `1px solid ${cragConf.border}` }}>
                        CRAG: {cragConf.label}
                    </span>
                )}

                {/* Score */}
                {pipeline.crag_top_score > 0 && (
                    <span className="text-[10px]" style={{ color: 'var(--apple-text-quaternary)' }}>
                        {pipeline.crag_top_score.toFixed(3)}
                    </span>
                )}

                {/* Results count */}
                <span className="text-[10px] ml-auto" style={{ color: 'var(--apple-text-quaternary)' }}>
                    {pipeline.total_candidates} &rarr; {pipeline.final_results}
                </span>

                {expanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
            </button>

            {/* Expanded stage timeline */}
            {expanded && pipelineStages.length > 0 && (
                <div className="mt-1 px-3.5 py-2.5 rounded-xl"
                    style={{
                        background: 'var(--apple-glass-bg)',
                        border: '1px solid var(--apple-glass-border)',
                        backdropFilter: 'blur(12px)',
                    }}>
                    <div>
                        {pipelineStages.map((stage, i) => (
                            <LiveStageRow key={stage.stage} stage={stage} isLast={i === pipelineStages.length - 1} />
                        ))}
                    </div>

                    {/* CRAG explanation */}
                    {pipeline.crag_confidence && pipeline.crag_confidence !== 'unknown' && (
                        <div className="mt-2 pt-2" style={{ borderTop: '1px solid var(--apple-divider)' }}>
                            <div className="flex items-start gap-2">
                                <ShieldIcon size={11} />
                                <p className="text-[10px]" style={{ color: 'var(--apple-text-tertiary)' }}>
                                    <span className="font-medium" style={{ color: 'var(--apple-text-secondary)' }}>CRAG: </span>
                                    {pipeline.crag_action === 'pass_through' && 'High-confidence retrieval — results used directly without modification.'}
                                    {pipeline.crag_action === 'refine_only' && 'Medium confidence — low-quality results filtered out, but no query rewrite needed.'}
                                    {pipeline.crag_action === 'hyde_rewrite' && 'Low confidence — a HyDE hypothetical document would be generated to improve retrieval.'}
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Source type breakdown */}
                    {pipeline.sources_by_type && Object.keys(pipeline.sources_by_type).length > 0 && (
                        <div className="mt-2 pt-2 flex items-center gap-2" style={{ borderTop: '1px solid var(--apple-divider)' }}>
                            <span className="text-[10px] font-medium" style={{ color: 'var(--apple-text-secondary)' }}>Sources:</span>
                            {Object.entries(pipeline.sources_by_type).map(([type, count]) => (
                                <span key={type} className="text-[9px] px-1.5 py-0.5 rounded-full"
                                    style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-tertiary)', border: '1px solid var(--apple-divider)' }}>
                                    {type}: {count}
                                </span>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
