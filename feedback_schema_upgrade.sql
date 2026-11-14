-- Run once in Supabase → SQL Editor (existing projects).
-- Adds anti-abuse fields used by feedback_memory.py

alter table public.insight_feedback
  add column if not exists trust double precision not null default 0.5;

alter table public.insight_feedback
  add column if not exists client_id text not null default '';

alter table public.insight_feedback
  add column if not exists text_hash text not null default '';

create index if not exists insight_feedback_client_ts_idx
  on public.insight_feedback (client_id, ts desc);

create index if not exists insight_feedback_text_hash_idx
  on public.insight_feedback (text_hash);

create index if not exists insight_feedback_trust_idx
  on public.insight_feedback (trust desc);
