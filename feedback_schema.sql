-- Run once in Supabase → SQL Editor
-- Then add to Streamlit secrets:
--   SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
--   SUPABASE_KEY = "YOUR_SERVICE_ROLE_OR_ANON_KEY"

create table if not exists public.insight_feedback (
  id text primary key,
  ts bigint not null,
  vote text not null check (vote in ('up', 'down')),
  text text not null,
  insight text not null,
  emotions jsonb not null default '[]'::jsonb,
  results_lang text not null default 'en',
  trust double precision not null default 0.5,
  client_id text not null default '',
  text_hash text not null default ''
);

create index if not exists insight_feedback_ts_idx
  on public.insight_feedback (ts desc);

create index if not exists insight_feedback_vote_idx
  on public.insight_feedback (vote);

-- If using the anon key, allow inserts/selects from the app:
alter table public.insight_feedback enable row level security;

drop policy if exists "insight_feedback_insert" on public.insight_feedback;
create policy "insight_feedback_insert"
  on public.insight_feedback
  for insert
  to anon, authenticated
  with check (true);

drop policy if exists "insight_feedback_select" on public.insight_feedback;
create policy "insight_feedback_select"
  on public.insight_feedback
  for select
  to anon, authenticated
  using (true);
