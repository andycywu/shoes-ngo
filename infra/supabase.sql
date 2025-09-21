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