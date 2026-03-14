import React, { useState, useRef, useEffect } from 'react'
import { formatAnswer } from '../utils/render'
import { SourceCard } from './SourceCard'
import { QualityBadges } from './metrics/QualityBadges'
import { QualityPanel } from './metrics/QualityPanel'
import { Tip } from './metrics/Tip'
import { PipelineInsight } from './PipelineInsight'
import { KnowledgeGraph } from './KnowledgeGraph'
import { AlertIcon, ZapIcon, CpuIcon, SearchIcon, ChevronDown, ChevronUp } from './Icons'

export function Message({ message, onViewSource }) {
    const [expandedSources, setExpandedSources] = useState({})
    const [showQuality, setShowQuality] = useState(false)
    const toggleSource = (idx) => setExpandedSources(p => ({ ...p, [idx]: !p[idx] }))
    const answerRef = useRef(null)

    // Handle clicks on [Source N] citation refs inside the rendered answer
    useEffect(() => {
        const el = answerRef.current
        if (!el || !onViewSource || !message.data?.sources?.length) return
        const handler = (e) => {
            const citEl = e.target.closest('[data-source-idx]')
            if (!citEl) return
            const idx = parseInt(citEl.dataset.sourceIdx, 10) - 1
            const source = message.data.sources[idx]
            if (source) onViewSource(source)
        }
        el.addEventListener('click', handler)
        return () => el.removeEventListener('click', handler)
    }, [message.data?.sources, onViewSource])

    if (message.role === 'user') {
        return (
            <div className="flex justify-end mb-5">
                <div className="px-5 py-3.5 rounded-[20px] rounded-br-md max-w-[75%] shadow-apple-sm" style={{ background: 'var(--apple-user-bubble)', color: '#ffffff' }}>
                    <p className="text-[14px] leading-relaxed whitespace-pre-wrap">{message.content}</p>
                </div>
            </div>
        )
    }

    const { data, streaming } = message
    const answerText = (message.content || '').split('\nSources:\n')[0]
    const quality = data?.quality
    const hasQuality = quality && quality.retrieval_confidence !== 'unknown' && !data?.deflected

    return (
        <div className="flex justify-start mb-7">
            <div className="max-w-[85%] w-full">
                {data?.deflected && !streaming && (
                    <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl mb-2 text-xs" style={{ background: 'var(--apple-yellow-bg)', border: '1px solid var(--apple-yellow-border)', color: 'var(--apple-yellow)' }}>
                        <AlertIcon size={14} />
                        <span>Deflected: {data.deflection_reason}</span>
                    </div>
                )}
                <div className="rounded-[20px] rounded-bl-md px-5 py-3.5" style={{ background: 'var(--apple-assistant-bubble)', backdropFilter: 'blur(20px) saturate(180%)', border: '1px solid var(--apple-glass-border)' }}>
                    <div ref={answerRef} className="text-[14px] leading-relaxed" style={{ color: 'var(--apple-text-primary)' }}>
                        {answerText ? formatAnswer(answerText) : (streaming ? '' : 'No response.')}
                        {streaming && <span className="inline-block w-2 h-4 ml-0.5 animate-pulse rounded-sm" style={{ background: 'var(--apple-accent)' }} />}
                    </div>
                    {data && !streaming && (
                        <div className="mt-3 pt-2 space-y-2" style={{ borderTop: '1px solid var(--apple-divider)' }}>
                            <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--apple-text-quaternary)' }}>
                                <Tip id="latency"><span className="flex items-center gap-1"><ZapIcon size={12} />{data.latency_ms?.toFixed(0) || '0'}ms</span></Tip>
                                <Tip id="backend_model"><span className="flex items-center gap-1"><CpuIcon size={12} />{data.backend || '...'}/{((data.model || '').split(':')[0]) || '...'}</span></Tip>
                                <Tip id="sources_count"><span className="flex items-center gap-1"><SearchIcon size={12} />{data.sources?.length || 0} relevant source{data.sources?.length !== 1 ? 's' : ''}</span></Tip>
                            </div>
                            {hasQuality && (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <QualityBadges quality={quality} />
                                    <button
                                        onClick={() => setShowQuality(v => !v)}
                                        style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--apple-text-quaternary)', display: 'flex', alignItems: 'center', gap: '2px', background: 'none', border: 'none', cursor: 'pointer', padding: '0' }}
                                        onMouseOver={e => e.currentTarget.style.color = 'var(--apple-text-primary)'}
                                        onMouseOut={e => e.currentTarget.style.color = 'var(--apple-text-quaternary)'}
                                    >
                                        {showQuality ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                                        <span>Details</span>
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                    {streaming && data?.sources?.length > 0 && (
                        <div className="flex items-center gap-2 mt-2 pt-2 text-xs" style={{ borderTop: '1px solid var(--apple-divider)', color: 'var(--apple-text-quaternary)' }}>
                            <SearchIcon size={12} />
                            <span>{data.sources.length} source{data.sources.length !== 1 ? 's' : ''} found — generating answer...</span>
                        </div>
                    )}
                </div>
                {showQuality && hasQuality && <QualityPanel quality={quality} />}
                {/* Live pipeline stages during streaming */}
                {streaming && data?.pipelineStages?.length > 0 && (
                    <PipelineInsight stages={data.pipelineStages} live={true} />
                )}
                {/* Final pipeline summary after streaming */}
                {!streaming && data?.pipeline && <PipelineInsight pipeline={data.pipeline} />}
                {data?.sources?.length > 0 && !data.deflected && (
                    <div className="mt-2 space-y-1.5">
                        {data.sources.map((source, idx) => (
                            <SourceCard key={idx} source={source} index={idx}
                                isExpanded={expandedSources[idx] || false}
                                onToggle={() => toggleSource(idx)}
                                onViewInPaper={onViewSource} />
                        ))}
                    </div>
                )}
                {/* Knowledge graph explorer — below sources, after streaming */}
                {!streaming && data?.sources?.length > 0 && !data.deflected && (
                    <KnowledgeGraph question={data.question || ''} />
                )}
            </div>
        </div>
    )
}
