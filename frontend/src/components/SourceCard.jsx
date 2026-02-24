import React from 'react'
import { EyeIcon, ChevronDown, ChevronUp } from './Icons'

export function SourceCard({ source, index, isExpanded, onToggle, onViewInPaper }) {
    const relevance = source.relevance || 'low'
    const badgeConfig = {
        high: { color: 'var(--apple-green)', bg: 'var(--apple-green-bg)', border: 'var(--apple-green-border)', label: 'High match' },
        medium: { color: 'var(--apple-yellow)', bg: 'var(--apple-yellow-bg)', border: 'var(--apple-yellow-border)', label: 'Partial match' },
        low: { color: 'var(--apple-orange)', bg: 'var(--apple-orange-bg)', border: 'var(--apple-orange-border)', label: 'Weak match' },
    }
    const badge = badgeConfig[relevance] || badgeConfig.low

    return (
        <div className="source-card glass-card p-4" style={{ borderColor: badge.border }}>
            <div className="flex items-start justify-between cursor-pointer" onClick={onToggle}>
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <span className="text-xs font-bold px-2 py-0.5 rounded-lg" style={{ color: 'var(--apple-accent)', background: 'var(--apple-accent-bg)' }}>
                            Source {index + 1}
                        </span>
                        <span className="text-xs font-mono font-semibold" style={{ color: badge.color }}>
                            {source.score.toFixed(2)}
                        </span>
                        <span className="text-[11px] px-1.5 py-0.5 rounded-lg" style={{ background: badge.bg, color: badge.color }}>
                            {badge.label}
                        </span>
                    </div>
                    <p className="text-sm mt-1 truncate font-medium" style={{ color: 'var(--apple-text-primary)' }}>{source.title}</p>
                    {source.section && <p className="text-xs mt-0.5" style={{ color: 'var(--apple-text-quaternary)' }}>{source.section}</p>}
                </div>
                <div className="flex items-center gap-1 ml-2 flex-shrink-0">
                    {source.paper_id && (
                        <button
                            onClick={(e) => { e.stopPropagation(); onViewInPaper && onViewInPaper(source) }}
                            className="p-1 rounded-lg transition-colors"
                            style={{ color: 'var(--apple-accent)' }}
                            title="View in paper">
                            <EyeIcon size={14} />
                        </button>
                    )}
                    <button style={{ color: 'var(--apple-text-quaternary)' }}>
                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                </div>
            </div>
            {isExpanded && (
                <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--apple-divider)' }}>
                    <p className="text-xs leading-relaxed whitespace-pre-wrap" style={{ color: 'var(--apple-text-secondary)' }}>
                        {source.text_preview}
                    </p>
                    {source.paper_id && (
                        <button
                            onClick={() => onViewInPaper && onViewInPaper(source)}
                            className="mt-2 text-xs flex items-center gap-1"
                            style={{ color: 'var(--apple-accent)' }}>
                            <EyeIcon size={12} /> View passage in full paper
                        </button>
                    )}
                </div>
            )}
        </div>
    )
}
