create table if not exists items (
  id uuid primary key default gen_random_uuid(),
  user_email text,
  image_url text,
  is_sneaker boolean,
  defects_json jsonb,
  suggestion text check (suggestion in ('resale','donate','recycle')),
  price_ranges_json jsonb,
  listing_title_zh text,
  listing_title_en text,
  listing_desc text,
  vlm_summary text,
  user_consent boolean default true,
  pii_blurred boolean default false,
  status text default 'created',
  created_at timestamptz default now()
);

create table if not exists logistics (
  id uuid primary key default gen_random_uuid(),
  item_id uuid references items(id) on delete cascade,
  route text check (route in ('RESALE','DONATE','RECYCLE')),
  qr_payload jsonb,
  scan_events_json jsonb default '[]'::jsonb,
  created_at timestamptz default now()
);

create table if not exists dataset_samples (
  id uuid primary key default gen_random_uuid(),
  item_id uuid references items(id) on delete cascade,
  image_path text,
  phash text,
  blur_score real,
  stage1_pred text,
  stage1_conf real,
  stage2_pred text,
  stage2_conf real,
  vlm_suggestion text,
  candidate_for_training boolean default false,
  is_labeled boolean default false,
  label_stage1 text,
  label_stage2 text,
  created_at timestamptz default now()
);

create table if not exists training_runs (
  id uuid primary key default gen_random_uuid(),
  started_at timestamptz default now(),
  finished_at timestamptz,
  status text check (status in ('running','succeeded','failed','canceled','pending_review')) default 'running',
  triggered_by text,                 -- 'auto' | 'manual:<user>'
  data_count integer,
  params_json jsonb,
  metrics_json jsonb,
  artifacts jsonb,
  error_msg text,
  gate_notes text,
  approved_at timestamptz,
  approved_by text
);

create table if not exists model_registry (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz default now(),
  model_name text,
  run_id uuid references training_runs(id) on delete set null,
  version text,
  metrics_json jsonb,
  artifacts jsonb,
  approved_by text
);

create table if not exists system_flags (
  key text primary key,
  value jsonb
);

create table if not exists users (
  id uuid primary key default gen_random_uuid(),
  email text unique not null,
  role text check (role in ('admin','staff','ngo','user')) not null default 'user',
  created_at timestamptz default now()
);

insert into system_flags(key, value) values
('auto_train_threshold', '{"min_new_samples":200,"min_label_ratio":0.7}')
on conflict (key) do nothing;

insert into system_flags(key, value) values
('cold_start_required', '{"enabled":true,"min_samples":80,"min_label_ratio":0.9}')
on conflict (key) do nothing;