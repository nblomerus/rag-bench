import React, { useState, useRef, useEffect, useCallback } from 'react'
import {
  Send, Loader2, BookOpen, AlertTriangle, ChevronDown, ChevronUp,
  Activity, Database, Cpu, BarChart3, Zap, Search, XCircle
} from 'lucide-react'

const API_BASE = '/api'

// ─── API helpers ──────────────────────────────────────────────────
async function queryRAG(question, topK = 5) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  })
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

async function fetchStats() {
  const res = await fetch(`${API_BASE}/stats`)
  if (!res.ok) throw new Error(`Stats error: ${res.status}`)
  return res.json()
}

async function runEval(runAll = false) {
  const res = await fetch(`${API_BASE}/eval`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_all: runAll }),
  })
  if (!res.ok) throw new Error(`Eval error: ${res.status}`)
  return res.json()
}

// ─── Citation formatter ───────────────────────────────────────────
function formatAnswer(text) {
  // Replace [Source N] with clickable spans
  return text.replace(
    /\[Source (\d+)\]/g,
    '<span class="citation-ref" data-source="$1">[Source $1]</span>'
  )
}

// ─── Source Card ──────────────────────────────────────────────────
function SourceCard({ source, index, isExpanded, onToggle }) {
  const scoreColor = source.score >= 5 ? 'text-green-400' :
                     source.score >= 3 ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className="source-card bg-gray-800 rounded-lg border border-gray-700 p-3">
      <div
        className="flex items-start justify-between cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-blue-400 bg-blue-400/10 px-2 py-0.5 rounded">
              Source {index + 1}
            </span>
            <span className={`text-xs font-mono ${scoreColor}`}>
              {source.score.toFixed(2)}
            </span>
          </div>
          <p className="text-sm text-gray-200 mt-1 truncate font-medium">
            {source.title}
          </p>
          {source.section && (
            <p className="text-xs text-gray-500 mt-0.5">{source.section}</p>
          )}
        </div>
        <button className="text-gray-500 hover:text-gray-300 ml-2 flex-shrink-0">
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>
      {isExpanded && (
        <div className="mt-2 pt-2 border-t border-gray-700">
          <p className="text-xs text-gray-400 leading-relaxed whitespace-pre-wrap">
            {source.text_preview}
          </p>
        </div>
      )}
    </div>
  )
}

// ─── Message Bubble ──────────────────────────────────────────────
function Message({ message }) {
  const [expandedSources, setExpandedSources] = useState({})

  const toggleSource = (idx) => {
    setExpandedSources(prev => ({ ...prev, [idx]: !prev[idx] }))
  }

  if (message.role === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="bg-blue-600 text-white px-4 py-3 rounded-2xl rounded-br-md max-w-[75%]">
          <p className="text-sm">{message.content}</p>
        </div>
      </div>
    )
  }

  // Assistant message
  const { data } = message

  return (
    <div className="flex justify-start mb-6">
      <div className="max-w-[85%] w-full">
        {/* Deflection banner */}
        {data?.deflected && (
          <div className="flex items-center gap-2 bg-amber-900/30 border border-amber-700/50 text-amber-300 px-3 py-2 rounded-lg mb-2 text-xs">
            <AlertTriangle size={14} />
            <span>Deflected: {data.deflection_reason}</span>
          </div>
        )}

        {/* Answer text */}
        <div className="bg-gray-800 border border-gray-700 rounded-2xl rounded-bl-md px-4 py-3">
          <div
            className="text-sm text-gray-200 leading-relaxed prose prose-invert max-w-none"
            dangerouslySetInnerHTML={{ __html: formatAnswer(message.content) }}
          />

          {/* Metadata bar */}
          {data && (
            <div className="flex items-center gap-3 mt-3 pt-2 border-t border-gray-700 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <Zap size={12} />
                {data.latency_ms?.toFixed(0)}ms
              </span>
              <span className="flex items-center gap-1">
                <Cpu size={12} />
                {data.backend}/{data.model?.split(':')[0]}
              </span>
              <span className="flex items-center gap-1">
                <Search size={12} />
                {data.sources?.length || 0} sources
              </span>
            </div>
          )}
        </div>

        {/* Source cards */}
        {data?.sources?.length > 0 && !data.deflected && (
          <div className="mt-2 space-y-1.5">
            {data.sources.map((source, idx) => (
              <SourceCard
                key={idx}
                source={source}
                index={idx}
                isExpanded={expandedSources[idx] || false}
                onToggle={() => toggleSource(idx)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Loading Indicator ───────────────────────────────────────────
function LoadingDots() {
  return (
    <div className="flex justify-start mb-4">
      <div className="bg-gray-800 border border-gray-700 rounded-2xl rounded-bl-md px-4 py-3">
        <div className="loading-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>
  )
}

// ─── Stats Panel ─────────────────────────────────────────────────
function StatsPanel({ stats, onClose }) {
  if (!stats) return null

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 mb-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
          <Database size={16} /> Pipeline Stats
        </h3>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
          <XCircle size={16} />
        </button>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {[
          { label: 'Papers', value: stats.total_papers, icon: BookOpen },
          { label: 'Chunks', value: stats.total_chunks?.toLocaleString(), icon: Database },
          { label: 'Embedding', value: stats.embedding_model?.split('/')[1], icon: Cpu },
          { label: 'LLM Backend', value: stats.llm_backend, icon: Activity },
          { label: 'Model', value: stats.llm_model?.split(':')[0], icon: Zap },
          { label: 'Collection', value: stats.collection_name, icon: Search },
        ].map(({ label, value, icon: Icon }) => (
          <div key={label} className="bg-gray-900 rounded-lg p-2">
            <div className="flex items-center gap-1.5 text-gray-500 text-xs mb-1">
              <Icon size={12} /> {label}
            </div>
            <p className="text-sm font-medium text-gray-200">{value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Eval Panel ──────────────────────────────────────────────────
function EvalPanel({ evalResults, isRunning, onRun, onClose }) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 mb-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
          <BarChart3 size={16} /> Evaluation Suite
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onRun(false)}
            disabled={isRunning}
            className="text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-3 py-1 rounded-md"
          >
            {isRunning ? 'Running...' : 'Quick Eval'}
          </button>
          <button
            onClick={() => onRun(true)}
            disabled={isRunning}
            className="text-xs bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-gray-200 px-3 py-1 rounded-md"
          >
            Full Eval
          </button>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
            <XCircle size={16} />
          </button>
        </div>
      </div>

      {isRunning && (
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Loader2 size={14} className="animate-spin" />
          Running evaluation queries...
        </div>
      )}

      {evalResults && (
        <>
          {/* Summary */}
          <div className="flex items-center gap-4 mb-3">
            <div className={`text-2xl font-bold ${
              evalResults.accuracy === 100 ? 'text-green-400' :
              evalResults.accuracy >= 90 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {evalResults.accuracy}%
            </div>
            <div className="text-xs text-gray-400">
              {evalResults.correct}/{evalResults.total} queries correct
            </div>
          </div>

          {/* By difficulty */}
          <div className="flex gap-2 mb-3">
            {Object.entries(evalResults.by_difficulty).map(([diff, data]) => (
              <span
                key={diff}
                className={`text-xs px-2 py-1 rounded-md ${
                  data.accuracy === 100 ? 'bg-green-900/30 text-green-400 border border-green-800' :
                  data.accuracy >= 80 ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-800' :
                  'bg-red-900/30 text-red-400 border border-red-800'
                }`}
              >
                {diff}: {data.correct}/{data.total}
              </span>
            ))}
          </div>

          {/* Results table */}
          <div className="max-h-60 overflow-y-auto space-y-1">
            {evalResults.results.map((r) => (
              <div
                key={r.id}
                className={`flex items-center gap-2 text-xs p-1.5 rounded ${
                  r.passed ? 'bg-gray-900' : 'bg-red-900/20'
                }`}
              >
                <span className={`font-bold ${r.passed ? 'text-green-400' : 'text-red-400'}`}>
                  {r.passed ? 'PASS' : 'FAIL'}
                </span>
                <span className="text-gray-500 font-mono w-16">{r.id}</span>
                <span className="text-gray-400 truncate flex-1">{r.question}</span>
                <span className="text-gray-500 font-mono">
                  {r.top_score.toFixed(1)}
                </span>
                {r.actual_deflect && (
                  <span className="text-amber-400 text-xs">
                    ({r.deflect_source})
                  </span>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// ─── Example Queries ─────────────────────────────────────────────
const EXAMPLE_QUERIES = [
  "What is the scaled dot-product attention formula?",
  "How does LoRA reduce trainable parameters?",
  "Compare RAG and RLHF approaches to factual accuracy",
  "What is the forward diffusion process in DDPM?",
  "What is the Mamba architecture's selective state space mechanism?",
]

// ─── Main App ────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [stats, setStats] = useState(null)
  const [showStats, setShowStats] = useState(false)
  const [showEval, setShowEval] = useState(false)
  const [evalResults, setEvalResults] = useState(null)
  const [evalRunning, setEvalRunning] = useState(false)
  const [error, setError] = useState(null)
  const [pipelineReady, setPipelineReady] = useState(false)

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Check health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`)
        const data = await res.json()
        setPipelineReady(data.pipeline_ready)
        if (!data.pipeline_ready) {
          setTimeout(checkHealth, 2000)
        }
      } catch {
        setTimeout(checkHealth, 3000)
      }
    }
    checkHealth()
  }, [])

  const handleSubmit = useCallback(async (questionOverride) => {
    const question = questionOverride || input.trim()
    if (!question || isLoading) return

    setInput('')
    setError(null)

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setIsLoading(true)

    try {
      const data = await queryRAG(question)

      // Split answer into text before "Sources:" section
      const answerParts = data.answer.split('\nSources:\n')
      const answerText = answerParts[0]

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: answerText,
        data: data,
      }])
    } catch (err) {
      setError(err.message)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}. Make sure the API server is running.`,
        data: null,
      }])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }, [input, isLoading])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleRunEval = async (runAll) => {
    setEvalRunning(true)
    try {
      const data = await runEval(runAll)
      setEvalResults(data)
    } catch (err) {
      setError(`Eval failed: ${err.message}`)
    } finally {
      setEvalRunning(false)
    }
  }

  const handleShowStats = async () => {
    if (!showStats) {
      try {
        const data = await fetchStats()
        setStats(data)
      } catch (err) {
        setError(`Stats failed: ${err.message}`)
      }
    }
    setShowStats(!showStats)
  }

  const isEmpty = messages.length === 0

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <BookOpen size={18} />
            </div>
            <div>
              <h1 className="text-sm font-bold">RAG-Bench</h1>
              <p className="text-xs text-gray-500">AI/ML Research Paper Assistant</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleShowStats}
              className={`text-xs px-3 py-1.5 rounded-md transition ${
                showStats ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'
              }`}
            >
              <Database size={12} className="inline mr-1" />
              Stats
            </button>
            <button
              onClick={() => setShowEval(!showEval)}
              className={`text-xs px-3 py-1.5 rounded-md transition ${
                showEval ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'
              }`}
            >
              <BarChart3 size={12} className="inline mr-1" />
              Eval
            </button>
            <div className={`w-2 h-2 rounded-full ${pipelineReady ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'}`} />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-4 flex flex-col">
        {/* Panels */}
        {showStats && <StatsPanel stats={stats} onClose={() => setShowStats(false)} />}
        {showEval && (
          <EvalPanel
            evalResults={evalResults}
            isRunning={evalRunning}
            onRun={handleRunEval}
            onClose={() => setShowEval(false)}
          />
        )}

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto pb-4">
          {/* Empty state */}
          {isEmpty && !isLoading && (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
              <div className="w-16 h-16 bg-blue-600/10 border border-blue-600/20 rounded-2xl flex items-center justify-center mb-4">
                <BookOpen size={28} className="text-blue-400" />
              </div>
              <h2 className="text-lg font-semibold text-gray-200 mb-1">
                Ask about AI/ML research
              </h2>
              <p className="text-sm text-gray-500 mb-6 max-w-md">
                Grounded answers with citations from {stats?.total_papers || '500+'} research papers.
                Every claim is backed by a source passage.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-w-lg w-full">
                {EXAMPLE_QUERIES.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => handleSubmit(q)}
                    disabled={!pipelineReady}
                    className="text-left text-xs bg-gray-800/50 hover:bg-gray-800 border border-gray-700/50
                               hover:border-gray-600 rounded-lg p-3 text-gray-400 hover:text-gray-200
                               transition disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {q}
                  </button>
                ))}
              </div>
              {!pipelineReady && (
                <p className="text-xs text-yellow-500 mt-4 flex items-center gap-1">
                  <Loader2 size={12} className="animate-spin" />
                  Loading pipeline... This may take a moment.
                </p>
              )}
            </div>
          )}

          {/* Message list */}
          {messages.map((msg, i) => (
            <Message key={i} message={msg} />
          ))}

          {/* Loading indicator */}
          {isLoading && <LoadingDots />}

          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="sticky bottom-0 pt-2 pb-2 bg-gray-950">
          {error && (
            <div className="text-xs text-red-400 mb-2 flex items-center gap-1">
              <AlertTriangle size={12} />
              {error}
            </div>
          )}
          <div className="flex items-end gap-2">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={pipelineReady ? "Ask a question about AI/ML research papers..." : "Pipeline loading..."}
                disabled={!pipelineReady || isLoading}
                rows={1}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 pr-12 text-sm
                           text-gray-200 placeholder-gray-500 resize-none focus:outline-none focus:ring-2
                           focus:ring-blue-500/50 focus:border-blue-500/50 disabled:opacity-50"
                style={{ minHeight: '48px', maxHeight: '120px' }}
                onInput={(e) => {
                  e.target.style.height = 'auto'
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
                }}
              />
            </div>
            <button
              onClick={() => handleSubmit()}
              disabled={!input.trim() || isLoading || !pipelineReady}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500
                         text-white p-3 rounded-xl transition flex-shrink-0"
            >
              {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
            </button>
          </div>
          <p className="text-[10px] text-gray-600 text-center mt-2">
            Answers are grounded in research papers. Always verify claims against original sources.
          </p>
        </div>
      </main>
    </div>
  )
}
