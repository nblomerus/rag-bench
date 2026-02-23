import React, { useState, useEffect } from 'react'
import { fetchStats, fetchBenchmarkHistory } from '../utils/api'
import { CpuIcon, BookIcon, DatabaseIcon, ZapIcon, SearchIcon, BarChartIcon, Spinner } from '../components/Icons'

export function ProductionTab() {
    const [stats, setStats] = useState(null)
    const [history, setHistory] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        let cancelled = false
        async function loadData() {
            setLoading(true)
            try {
                const [statsData, historyData] = await Promise.allSettled([
                    fetchStats(),
                    fetchBenchmarkHistory().catch(() => null),
                ])
                if (cancelled) return
                if (statsData.status === 'fulfilled') setStats(statsData.value)
                if (historyData.status === 'fulfilled') setHistory(historyData.value)
            } catch (err) {
                if (!cancelled) setError(err.message)
            } finally {
                if (!cancelled) setLoading(false)
            }
        }
        loadData()
        return () => { cancelled = true }
    }, [])

    if (loading) {
        return (
            <div className="flex-1 flex items-center justify-center">
                <div className="flex items-center gap-2 text-sm text-gray-400">
                    <Spinner size={16} /> Loading production info...
                </div>
            </div>
        )
    }

    return (
        <div className="flex-1 overflow-y-auto">
            <div className="max-w-4xl mx-auto px-4 py-6">
                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 bg-blue-600/10 border border-blue-600/20 rounded-xl flex items-center justify-center">
                        <CpuIcon size={20} />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-gray-100">Production</h2>
                        <p className="text-xs text-gray-500">Pipeline configuration, system health, and deployment info</p>
                    </div>
                </div>

                {error && (
                    <div className="text-sm text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-4 py-3 mb-6">
                        {error}
                    </div>
                )}

                {/* Pipeline Configuration */}
                {stats && (
                    <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 mb-6">
                        <h3 className="text-sm font-semibold text-gray-200 mb-4 flex items-center gap-2">
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
                                <div key={label} className="bg-gray-900 rounded-lg p-3">
                                    <div className="flex items-center gap-1.5 text-gray-500 text-xs mb-1">
                                        <Ic size={12} /> {label}
                                    </div>
                                    <p className="text-sm font-medium text-gray-200">{value}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* System Health */}
                <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 mb-6">
                    <h3 className="text-sm font-semibold text-gray-200 mb-4 flex items-center gap-2">
                        <ZapIcon size={16} /> System Health
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="bg-gray-900 rounded-lg p-3">
                            <div className="text-xs text-gray-500 mb-1">API Status</div>
                            <div className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                                <span className="text-sm text-green-400">Online</span>
                            </div>
                        </div>
                        <div className="bg-gray-900 rounded-lg p-3">
                            <div className="text-xs text-gray-500 mb-1">Pipeline Ready</div>
                            <div className="flex items-center gap-2">
                                <span className={`w-2 h-2 ${stats ? 'bg-green-500' : 'bg-yellow-500'} rounded-full`}></span>
                                <span className={`text-sm ${stats ? 'text-green-400' : 'text-yellow-400'}`}>
                                    {stats ? 'Ready' : 'Loading'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Evaluation History */}
                <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
                    <h3 className="text-sm font-semibold text-gray-200 mb-4 flex items-center gap-2">
                        <BarChartIcon size={16} /> Evaluation History
                    </h3>
                    {history && history.runs && history.runs.length > 0 ? (
                        <div className="space-y-2">
                            {history.runs.map((run, i) => (
                                <div key={i} className="flex items-center gap-3 text-xs bg-gray-900 rounded-lg p-2.5">
                                    <span className="text-gray-500 font-mono">{run.timestamp}</span>
                                    <span className="text-blue-400">{run.benchmark}</span>
                                    <span className="text-gray-400 flex-1">{run.total_evaluated} entries</span>
                                    {run.accuracy !== undefined && (
                                        <span className={`font-bold ${run.accuracy >= 0.8 ? 'text-green-400' : run.accuracy >= 0.6 ? 'text-yellow-400' : 'text-red-400'}`}>
                                            {(run.accuracy * 100).toFixed(1)}%
                                        </span>
                                    )}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-xs text-gray-600">No evaluation runs recorded yet. Run a benchmark from the Benchmarks tab to see history here.</p>
                    )}
                </div>
            </div>
        </div>
    )
}
