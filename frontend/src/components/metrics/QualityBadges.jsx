import React from 'react'
import { Tip } from './Tip'

export function badgeStyle(level) {
    if (level === 'green') return { background: '#052e16', color: '#4ade80', borderColor: '#166534' }
    if (level === 'yellow') return { background: '#422006', color: '#fbbf24', borderColor: '#92400e' }
    return { background: '#450a0a', color: '#f87171', borderColor: '#991b1b' }
}

export function QualityBadges({ quality }) {
    if (!quality) return null
    const conf = quality.retrieval_confidence || 'unknown'
    const confLevel = conf === 'high' ? 'green' : conf === 'medium' ? 'yellow' : 'red'
    const confDot = conf === 'high' ? '#4ade80' : conf === 'medium' ? '#fbbf24' : '#f87171'

    const perSource = quality.per_source_cited || []
    const anyFooterOnly = perSource.some(s => s.footer_only)
    const cov = quality.citation_coverage || 0
    const covLevel = cov === 0 ? 'red' : anyFooterOnly ? 'yellow' : cov > 0.8 ? 'green' : cov > 0.5 ? 'yellow' : 'red'

    const unsup = quality.unsupported_claims || 0
    const unsupLevel = unsup === 0 ? 'green' : unsup <= 2 ? 'yellow' : 'red'

    const faith = quality.faithfulness_score || 0
    const faithLevel = faith >= 4 ? 'green' : faith >= 3 ? 'yellow' : 'red'

    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <Tip id="retrieval_confidence">
                <span className="quality-badge" style={badgeStyle(confLevel)}>
                    <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: confDot, display: 'inline-block' }}></span>
                    {conf === 'unknown' ? 'N/A' : conf.charAt(0).toUpperCase() + conf.slice(1)}
                </span>
            </Tip>
            <Tip id="sources_cited">
                <span className="quality-badge" style={badgeStyle(covLevel)} title={anyFooterOnly ? 'Citations found in footer block (not inline)' : ''}>
                    {quality.sources_cited}/{quality.sources_provided} cited{anyFooterOnly ? ' \u2020' : ''}
                </span>
            </Tip>
            <Tip id="unsupported_claims">
                <span className="quality-badge" style={badgeStyle(unsupLevel)}>
                    {unsup} unsupported
                </span>
            </Tip>
            {faith > 0 && (
                <Tip id="faithfulness">
                    <span className="quality-badge" style={badgeStyle(faithLevel)}>
                        Faith {faith.toFixed(1)}/5
                    </span>
                </Tip>
            )}
        </div>
    )
}
