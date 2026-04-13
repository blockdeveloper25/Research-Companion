/**
 * EntityBrowser.tsx — Browse and explore the knowledge graph.
 *
 * Three sections:
 *   1. Stats bar    — total entities, relationships, breakdown by type
 *   2. Filter bar   — search by name, filter by entity type
 *   3. Entity list  — clickable rows that expand to show relationships
 *
 * This is a debugging / exploration tool — it lets you verify that
 * graph extraction is working correctly before trusting it to improve answers.
 */

import { useCallback, useEffect, useState } from 'react'
import { Search, ChevronDown, ChevronRight, ArrowLeft, ArrowRight } from 'lucide-react'
import { clsx } from 'clsx'
import * as api from '../lib/api'
import type { GraphEntity, GraphRelationship, GraphStats } from '../types'

// Color-coded badges for each entity type — makes scanning the list fast.
const TYPE_COLORS: Record<string, string> = {
  PERSON:       'bg-blue-100 text-blue-700',
  POLICY:       'bg-purple-100 text-purple-700',
  DEPARTMENT:   'bg-green-100 text-green-700',
  ORGANIZATION: 'bg-orange-100 text-orange-700',
  ROLE:         'bg-yellow-100 text-yellow-700',
  DOCUMENT:     'bg-pink-100 text-pink-700',
  DATE:         'bg-gray-100 text-gray-600',
  AMOUNT:       'bg-emerald-100 text-emerald-700',
  LOCATION:     'bg-cyan-100 text-cyan-700',
  OTHER:        'bg-gray-100 text-gray-500',
}

export default function EntityBrowser() {
  const [stats, setStats]         = useState<GraphStats | null>(null)
  const [entities, setEntities]   = useState<GraphEntity[]>([])
  const [search, setSearch]       = useState('')
  const [typeFilter, setTypeFilter] = useState('')
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [loading, setLoading]     = useState(true)

  // ── Load stats on mount ────────────────────────────────────────────────────

  useEffect(() => {
    api.getGraphStats().then(setStats).catch(() => {})
  }, [])

  // ── Load entities when filters change ──────────────────────────────────────

  useEffect(() => {
    setLoading(true)
    const timer = setTimeout(() => {
      api.listEntities({
        search: search || undefined,
        type: typeFilter || undefined,
        limit: 100,
      })
        .then(setEntities)
        .catch(() => setEntities([]))
        .finally(() => setLoading(false))
    }, 300) // debounce search input

    return () => clearTimeout(timer)
  }, [search, typeFilter])

  // ── Toggle entity expansion ────────────────────────────────────────────────

  const handleToggle = useCallback((id: string) => {
    setExpandedId(prev => prev === id ? null : id)
  }, [])

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full bg-surface">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="px-6 pt-6 pb-4 border-b border-border">
        <h1 className="text-lg font-semibold tracking-tight">Knowledge Graph</h1>
        {stats && (
          <div className="flex gap-4 mt-2 text-xs text-muted">
            <span>{stats.total_entities} entities</span>
            <span>{stats.total_relationships} relationships</span>
            <span>{stats.total_documents_with_metadata} documents</span>
          </div>
        )}
      </div>

      {/* ── Stats breakdown by type ────────────────────────────────────────── */}
      {stats && stats.total_entities > 0 && (
        <div className="flex flex-wrap gap-1.5 px-6 pt-3 pb-2">
          {Object.entries(stats.entities_by_type).map(([type, count]) => (
            <button
              key={type}
              onClick={() => setTypeFilter(prev => prev === type ? '' : type)}
              className={clsx(
                'px-2 py-0.5 rounded-full text-xs transition-all',
                typeFilter === type
                  ? 'ring-1 ring-black/30 font-medium'
                  : 'opacity-70 hover:opacity-100',
                TYPE_COLORS[type] || TYPE_COLORS.OTHER,
              )}
            >
              {type} ({count})
            </button>
          ))}
        </div>
      )}

      {/* ── Search bar ─────────────────────────────────────────────────────── */}
      <div className="px-6 py-3">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-dim" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search entities..."
            className="w-full pl-9 pr-3 py-2 text-sm bg-raised border border-border rounded-lg
                       outline-none focus:ring-1 focus:ring-black/20 placeholder:text-dim"
          />
        </div>
      </div>

      {/* ── Entity list ────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 pb-6">
        {loading ? (
          <p className="text-sm text-muted text-center mt-8">Loading...</p>
        ) : entities.length === 0 ? (
          <div className="text-center mt-8">
            <p className="text-sm text-muted">No entities found</p>
            <p className="text-xs text-dim mt-1">
              {stats?.total_entities === 0
                ? 'Ingest documents with --force to populate the graph'
                : 'Try adjusting your search or filters'}
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {entities.map(entity => (
              <EntityRow
                key={entity.id}
                entity={entity}
                isExpanded={expandedId === entity.id}
                onToggle={() => handleToggle(entity.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}


// ── Entity row with expandable relationships ────────────────────────────────

function EntityRow({
  entity,
  isExpanded,
  onToggle,
}: {
  entity:     GraphEntity
  isExpanded: boolean
  onToggle:   () => void
}) {
  const [relationships, setRelationships] = useState<GraphRelationship[]>([])
  const [loadingRels, setLoadingRels]     = useState(false)

  // Load relationships when expanded
  useEffect(() => {
    if (!isExpanded) return
    setLoadingRels(true)
    api.getEntity(entity.id)
      .then(data => setRelationships(data.relationships))
      .catch(() => setRelationships([]))
      .finally(() => setLoadingRels(false))
  }, [isExpanded, entity.id])

  const typeColor = TYPE_COLORS[entity.entity_type] || TYPE_COLORS.OTHER

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      {/* ── Row header ───────────────────────────────────────────────────── */}
      <button
        onClick={onToggle}
        className={clsx(
          'w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors',
          isExpanded ? 'bg-raised' : 'hover:bg-raised/50',
        )}
      >
        {isExpanded
          ? <ChevronDown size={14} className="text-muted shrink-0" />
          : <ChevronRight size={14} className="text-muted shrink-0" />
        }

        <span className={clsx('px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0', typeColor)}>
          {entity.entity_type}
        </span>

        <span className="text-sm font-medium truncate flex-1">{entity.name}</span>

        <span className="text-[11px] text-dim shrink-0 truncate max-w-[140px]">
          {entity.source}, p.{entity.page}
        </span>
      </button>

      {/* ── Expanded detail ──────────────────────────────────────────────── */}
      {isExpanded && (
        <div className="px-4 py-3 border-t border-border bg-white">
          {entity.description && (
            <p className="text-xs text-muted mb-3">{entity.description}</p>
          )}

          {loadingRels ? (
            <p className="text-xs text-dim">Loading relationships...</p>
          ) : relationships.length === 0 ? (
            <p className="text-xs text-dim">No relationships found</p>
          ) : (
            <div className="flex flex-col gap-1.5">
              <p className="text-[10px] font-medium uppercase tracking-widest text-muted mb-1">
                Relationships
              </p>
              {relationships.map(rel => (
                <RelationshipRow
                  key={rel.id}
                  rel={rel}
                  currentEntityId={entity.id}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}


// ── Single relationship row ─────────────────────────────────────────────────

function RelationshipRow({
  rel,
  currentEntityId,
}: {
  rel:             GraphRelationship
  currentEntityId: string
}) {
  // Determine direction relative to the current entity
  const isOutgoing = rel.source_entity_id === currentEntityId
  const otherName = isOutgoing ? (rel.target_name || '?') : (rel.source_name || '?')
  const otherType = isOutgoing ? (rel.target_type || '') : (rel.source_type || '')
  const typeColor = TYPE_COLORS[otherType] || TYPE_COLORS.OTHER

  return (
    <div className="flex items-center gap-2 text-xs py-1">
      {isOutgoing
        ? <ArrowRight size={12} className="text-muted shrink-0" />
        : <ArrowLeft size={12} className="text-muted shrink-0" />
      }
      <span className="text-muted font-medium shrink-0">{rel.relation_type}</span>
      {otherType && (
        <span className={clsx('px-1 py-0.5 rounded text-[9px]', typeColor)}>
          {otherType}
        </span>
      )}
      <span className="truncate">{otherName}</span>
      {rel.description && (
        <span className="text-dim truncate ml-auto">— {rel.description}</span>
      )}
    </div>
  )
}
