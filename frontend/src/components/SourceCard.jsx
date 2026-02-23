import React from 'react'
import { EyeIcon, ChevronDown, ChevronUp } from './Icons'

export function SourceCard({ source, index, isExpanded, onToggle, onViewInPaper }) {
    const relevance = source.relevance || 'low'
    const badgeConfig = {
        high: { color: 'text-green-400', bg: 'bg-green-400/10', border: 'border-green-800/50', label: 'High match' },
        medium: { color: 'text-yellow-400', bg: 'bg-yellow-400/10', border: 'border-yellow-800/50', label: 'Partial match' },
        low: { color: 'text-orange-400', bg: 'bg-orange-400/10', border: 'border-orange-800/50', label: 'Weak match' },
    }
    const badge = badgeConfig[relevance] || badgeConfig.low

    return (
        <div className={`source-card bg-gray-800 rounded-lg border ${badge.border} p-3`}>
            <div className="flex items-start justify-between cursor-pointer" onClick={onToggle}>
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-blue-400 bg-blue-400/10 px-2 py-0.5 rounded">
                            Source {index + 1}
                        </span>
                        <span className={`text-xs font-mono font-semibold ${badge.color}`}>
                            {source.score.toFixed(2)}
                        </span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${badge.bg} ${badge.color}`}>
                            {badge.label}
                        </span>
                    </div>
                    <p className="text-sm text-gray-200 mt-1 truncate font-medium">{source.title}</p>
                    {source.section && <p className="text-xs text-gray-500 mt-0.5">{source.section}</p>}
                </div>
                <div className="flex items-center gap-1 ml-2 flex-shrink-0">
                    {source.paper_id && (
                        <button
                            onClick={(e) => { e.stopPropagation(); onViewInPaper && onViewInPaper(source) }}
                            className="text-blue-400 hover:text-blue-300 p-1 rounded hover:bg-blue-400/10 transition"
                            title="View in paper">
                            <EyeIcon size={14} />
                        </button>
                    )}
                    <button className="text-gray-500 hover:text-gray-300">
                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                </div>
            </div>
            {isExpanded && (
                <div className="mt-2 pt-2 border-t border-gray-700">
                    <p className="text-xs text-gray-400 leading-relaxed whitespace-pre-wrap">
                        {source.text_preview}
                    </p>
                    {source.paper_id && (
                        <button
                            onClick={() => onViewInPaper && onViewInPaper(source)}
                            className="mt-2 text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
                            <EyeIcon size={12} /> View passage in full paper
                        </button>
                    )}
                </div>
            )}
        </div>
    )
}
