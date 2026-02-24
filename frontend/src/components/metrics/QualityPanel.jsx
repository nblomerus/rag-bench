import React from 'react'
import { Tip } from './Tip'
import { ScoreBar } from './ScoreBar'

export function QualityPanel({ quality }) {
    if (!quality) return null
    const spread = quality.score_spread || {}
    const diversity = quality.source_diversity || {}
    const perSource = quality.per_source_cited || []
    const maxScore = spread.max || 1
    const conf = quality.retrieval_confidence || 'unknown'
    const faith = quality.faithfulness_score || 0

    return (
        <div className="glass-card p-4 mt-2">
            {/* Retrieval */}
            <div style={{ marginBottom: '12px' }}>
                <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--apple-text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>Retrieval</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '12px' }}>
                    <div>
                        <Tip id="retrieval_confidence">
                            <span style={{ color: 'var(--apple-text-tertiary)' }}>Confidence: </span>
                            <span style={{ color: conf === 'high' ? 'var(--apple-green)' : conf === 'medium' ? 'var(--apple-yellow)' : 'var(--apple-red)' }}>{conf}</span>
                            <span style={{ color: 'var(--apple-text-quaternary)', marginLeft: '4px' }}>(top: {(quality.top_retrieval_score || 0).toFixed(2)})</span>
                        </Tip>
                    </div>
                    <div>
                        <Tip id="diversity">
                            <span style={{ color: 'var(--apple-text-tertiary)' }}>Diversity: </span>
                            <span style={{ color: 'var(--apple-text-primary)' }}>{diversity.unique_papers || 0} papers, {diversity.unique_sections || 0} sections</span>
                        </Tip>
                    </div>
                    {spread.gap_ratio > 0 && (
                        <div>
                            <Tip id="score_gap">
                                <span style={{ color: 'var(--apple-text-tertiary)' }}>Score gap: </span>
                                <span style={{ color: spread.gap_ratio >= 1.8 ? 'var(--apple-green)' : spread.gap_ratio >= 1.3 ? 'var(--apple-yellow)' : 'var(--apple-text-tertiary)' }}>
                                    {spread.gap_ratio.toFixed(1)}\u00d7
                                </span>
                                <span style={{ color: 'var(--apple-text-quaternary)', marginLeft: '4px' }}>vs #2</span>
                            </Tip>
                        </div>
                    )}
                </div>
                {perSource.length > 0 && (
                    <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {perSource.map((s, i) => (
                            <Tip key={i} id="score_bar">
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                                    <span style={{ color: 'var(--apple-text-tertiary)', width: '16px', textAlign: 'right' }}>{s.source_number}</span>
                                    <ScoreBar value={s.score} max={maxScore} color={s.score >= maxScore * 0.7 ? 'green' : s.score >= maxScore * 0.4 ? 'yellow' : 'red'} />
                                    <span style={{ color: 'var(--apple-text-tertiary)', fontFamily: 'monospace', width: '40px' }}>{(s.score || 0).toFixed(1)}</span>
                                </div>
                            </Tip>
                        ))}
                    </div>
                )}
            </div>

            {/* Citations */}
            <div style={{ marginBottom: '12px' }}>
                <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--apple-text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>Citations</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', fontSize: '12px' }}>
                    <div><Tip id="coverage"><span style={{ color: 'var(--apple-text-tertiary)' }}>Coverage: </span><span style={{ color: 'var(--apple-text-primary)' }}>{((quality.citation_coverage || 0) * 100).toFixed(0)}%</span></Tip></div>
                    <div><Tip id="density"><span style={{ color: 'var(--apple-text-tertiary)' }}>Density: </span><span style={{ color: 'var(--apple-text-primary)' }}>{(quality.citation_density || 0).toFixed(1)}/sentence</span></Tip></div>
                    <div><Tip id="unsupported_claims"><span style={{ color: 'var(--apple-text-tertiary)' }}>Unsupported: </span><span style={{ color: (quality.unsupported_claims || 0) === 0 ? 'var(--apple-green)' : 'var(--apple-yellow)' }}>{quality.unsupported_claims || 0}</span></Tip></div>
                </div>
            </div>

            {/* Faithfulness */}
            {faith > 0 && (
                <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--apple-text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>Faithfulness</div>
                    <Tip id="faithfulness">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px' }}>
                            <ScoreBar value={faith} max={5} color={faith >= 4 ? 'green' : faith >= 3 ? 'yellow' : 'red'} />
                            <span style={{ color: faith >= 4 ? 'var(--apple-green)' : faith >= 3 ? 'var(--apple-yellow)' : 'var(--apple-red)' }}>{faith.toFixed(1)} / 5</span>
                            <span style={{ color: 'var(--apple-text-quaternary)' }}>keyword overlap with sources</span>
                        </div>
                    </Tip>
                </div>
            )}

            {/* Source Verification */}
            {perSource.length > 0 && (
                <div>
                    <Tip id="source_verification">
                        <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--apple-text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>Source Verification</div>
                    </Tip>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {perSource.map((s, i) => (
                            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                                <span style={{ color: 'var(--apple-text-tertiary)', width: '16px', textAlign: 'right' }}>{s.source_number}</span>
                                <span style={{ color: 'var(--apple-text-secondary)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={s.title}>{s.title}</span>
                                <span style={{ color: 'var(--apple-text-tertiary)', fontFamily: 'monospace', width: '40px' }}>{(s.score || 0).toFixed(2)}</span>
                                <span style={{ color: s.cited ? (s.footer_only ? 'var(--apple-yellow)' : 'var(--apple-green)') : 'var(--apple-text-quaternary)' }} title={s.footer_only ? 'Cited in footer block only' : ''}>
                                    {s.cited ? (s.footer_only ? '\u2713 footer' : `\u2713 ${s.citation_count}\u00d7`) : '\u2717'}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
