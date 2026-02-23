import React from 'react'
import { BarChartIcon, XIcon, Spinner } from './Icons'

export function EvalPanel({ evalResults, isRunning, onRun, onClose }) {
    return (
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 mb-4">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                    <BarChartIcon size={16} /> Evaluation Suite
                </h3>
                <div className="flex items-center gap-2">
                    <button onClick={() => onRun(false)} disabled={isRunning}
                        className="text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-3 py-1 rounded-md">
                        {isRunning ? 'Running...' : 'Quick Eval'}
                    </button>
                    <button onClick={() => onRun(true)} disabled={isRunning}
                        className="text-xs bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-gray-200 px-3 py-1 rounded-md">
                        Full Eval
                    </button>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-300"><XIcon size={16} /></button>
                </div>
            </div>
            {isRunning && (
                <div className="flex items-center gap-2 text-sm text-gray-400">
                    <Spinner size={14} /> Running evaluation queries...
                </div>
            )}
            {evalResults && (
                <div>
                    <div className="flex items-center gap-4 mb-3">
                        <div className={`text-2xl font-bold ${evalResults.accuracy === 100 ? 'text-green-400' :
                            evalResults.accuracy >= 90 ? 'text-yellow-400' : 'text-red-400'
                            }`}>{evalResults.accuracy}%</div>
                        <div className="text-xs text-gray-400">{evalResults.correct}/{evalResults.total} correct</div>
                    </div>
                    <div className="flex gap-2 mb-3 flex-wrap">
                        {Object.entries(evalResults.by_difficulty).map(([diff, data]) => (
                            <span key={diff} className={`text-xs px-2 py-1 rounded-md border ${data.accuracy === 100 ? 'bg-green-900/30 text-green-400 border-green-800' :
                                data.accuracy >= 80 ? 'bg-yellow-900/30 text-yellow-400 border-yellow-800' :
                                    'bg-red-900/30 text-red-400 border-red-800'
                                }`}>{diff}: {data.correct}/{data.total}</span>
                        ))}
                    </div>
                    <div className="max-h-60 overflow-y-auto space-y-1">
                        {evalResults.results.map((r) => (
                            <div key={r.id} className={`flex items-center gap-2 text-xs p-1.5 rounded ${r.passed ? 'bg-gray-900' : 'bg-red-900/20'}`}>
                                <span className={`font-bold ${r.passed ? 'text-green-400' : 'text-red-400'}`}>
                                    {r.passed ? 'PASS' : 'FAIL'}
                                </span>
                                <span className="text-gray-500 font-mono w-16">{r.id}</span>
                                <span className="text-gray-400 truncate flex-1">{r.question}</span>
                                <span className="text-gray-500 font-mono">{r.top_score.toFixed(1)}</span>
                                {r.actual_deflect && (
                                    <span className="text-amber-400 text-xs">({r.deflect_source})</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
