import React from 'react'
import { BarChartIcon, XIcon, Spinner } from './Icons'

export function EvalPanel({ evalResults, isRunning, onRun, onClose }) {
    return (
        <div className="glass-card p-5 mb-4">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                    <BarChartIcon size={16} /> Evaluation Suite
                </h3>
                <div className="flex items-center gap-2">
                    <button onClick={() => onRun(false)} disabled={isRunning}
                        className="text-xs disabled:opacity-50 text-white px-3 py-1.5 rounded-lg transition-colors"
                        style={{ background: 'var(--apple-accent)' }}>
                        {isRunning ? 'Running...' : 'Quick Eval'}
                    </button>
                    <button onClick={() => onRun(true)} disabled={isRunning}
                        className="text-xs disabled:opacity-50 px-3 py-1.5 rounded-lg transition-colors"
                        style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-primary)' }}>
                        Full Eval
                    </button>
                    <button onClick={onClose} style={{ color: 'var(--apple-text-quaternary)' }} className="hover:opacity-80"><XIcon size={16} /></button>
                </div>
            </div>
            {isRunning && (
                <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--apple-text-secondary)' }}>
                    <Spinner size={14} /> Running evaluation queries...
                </div>
            )}
            {evalResults && (
                <div>
                    <div className="flex items-center gap-4 mb-3">
                        <div className="text-2xl font-bold" style={{ color: evalResults.accuracy === 100 ? 'var(--apple-green)' : evalResults.accuracy >= 90 ? 'var(--apple-yellow)' : 'var(--apple-red)' }}>
                            {evalResults.accuracy}%
                        </div>
                        <div className="text-xs" style={{ color: 'var(--apple-text-secondary)' }}>{evalResults.correct}/{evalResults.total} correct</div>
                    </div>
                    <div className="flex gap-2 mb-3 flex-wrap">
                        {Object.entries(evalResults.by_difficulty).map(([diff, data]) => (
                            <span key={diff} className="text-xs px-2.5 py-1 rounded-lg border" style={{
                                background: data.accuracy === 100 ? 'var(--apple-green-bg)' : data.accuracy >= 80 ? 'var(--apple-yellow-bg)' : 'var(--apple-red-bg)',
                                color: data.accuracy === 100 ? 'var(--apple-green)' : data.accuracy >= 80 ? 'var(--apple-yellow)' : 'var(--apple-red)',
                                borderColor: data.accuracy === 100 ? 'var(--apple-green-border)' : data.accuracy >= 80 ? 'var(--apple-yellow-border)' : 'var(--apple-red-border)',
                            }}>{diff}: {data.correct}/{data.total}</span>
                        ))}
                    </div>
                    <div className="max-h-60 overflow-y-auto space-y-1">
                        {evalResults.results.map((r) => (
                            <div key={r.id} className="flex items-center gap-2 text-xs p-2 rounded-lg" style={{ background: r.passed ? 'var(--apple-bg-tertiary)' : 'var(--apple-red-bg)' }}>
                                <span className="font-bold" style={{ color: r.passed ? 'var(--apple-green)' : 'var(--apple-red)' }}>
                                    {r.passed ? 'PASS' : 'FAIL'}
                                </span>
                                <span className="font-mono w-16" style={{ color: 'var(--apple-text-quaternary)' }}>{r.id}</span>
                                <span className="truncate flex-1" style={{ color: 'var(--apple-text-secondary)' }}>{r.question}</span>
                                <span className="font-mono" style={{ color: 'var(--apple-text-quaternary)' }}>{r.top_score.toFixed(1)}</span>
                                {r.actual_deflect && (
                                    <span className="text-xs" style={{ color: 'var(--apple-yellow)' }}>({r.deflect_source})</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
