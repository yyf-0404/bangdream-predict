import unittest

from data.cutoff_sources import (
    cutoff_source_urls,
    fetch_and_merge_cutoffs,
    filter_terminal_drop,
    filter_starfx_outliers,
    merge_cutoff_sources,
    normalize_cutoff_points,
)
from data.raw import prepare_pchip_points


T0 = 1_700_000_000_000
STEP = 30 * 60 * 1000


class CutoffSourceTests(unittest.TestCase):
    def test_terminal_drop_is_removed_but_normal_end_growth_is_kept(self):
        dropping = [
            {"time": T0, "ep": 100},
            {"time": T0 + STEP, "ep": 120},
            {"time": T0 + 2 * STEP, "ep": 110},
        ]
        growing = [
            {"time": T0, "ep": 100},
            {"time": T0 + STEP, "ep": 120},
            {"time": T0 + 2 * STEP, "ep": 130},
        ]

        cleaned, report = filter_terminal_drop(dropping, T0 + 2 * STEP)
        kept, kept_report = filter_terminal_drop(growing, T0 + 2 * STEP)

        self.assertEqual(cleaned, dropping[:-1])
        self.assertEqual(report["removed"], 1)
        self.assertEqual(report["rejected"][0]["reason"], "terminal_settlement_drop")
        self.assertEqual(kept, growing)
        self.assertEqual(kept_report["removed"], 0)

        # 即使尾部发生下降，只要它不是精确 endAt，也不适用结算特例。
        in_event, in_event_report = filter_terminal_drop(
            dropping, T0 + 3 * STEP
        )
        self.assertEqual(in_event, dropping)
        self.assertEqual(in_event_report["removed"], 0)

    def test_pchip_boundaries_forward_fill_nonempty_but_keep_empty_zero(self):
        points = prepare_pchip_points(
            [{"time": T0 + STEP, "ep": 123}],
            T0,
            T0 + 2 * STEP,
        )
        empty = prepare_pchip_points([], T0, T0 + 2 * STEP)

        self.assertEqual(points, [
            {"time": T0, "ep": 0},
            {"time": T0 + STEP, "ep": 123},
            {"time": T0 + 2 * STEP, "ep": 123},
        ])
        self.assertEqual(empty, [
            {"time": T0, "ep": 0},
            {"time": T0 + 2 * STEP, "ep": 0},
        ])

    def test_fetch_removes_terminal_drop_before_starfx_monotonic_cleaning(self):
        def fake_fetch(url):
            if "hhwx.org" in url:
                end_ep = 130  # HHWX 仍是正常赛中增长，应该保留。
            else:
                end_ep = 110  # Bestdori/StarFX 的封禁结算下降。
            return {"cutoffs": [
                {"time": T0, "ep": 100},
                {"time": T0 + STEP, "ep": 120},
                {"time": T0 + 2 * STEP, "ep": end_ep},
            ]}

        merged, report = fetch_and_merge_cutoffs(
            3, 999, 1000,
            start_at=T0,
            end_at=T0 + 2 * STEP,
            fetch_json=fake_fetch,
        )

        self.assertEqual(merged, [
            {"time": T0, "ep": 100},
            {"time": T0 + STEP, "ep": 120},
            {"time": T0 + 2 * STEP, "ep": 130},
        ])
        self.assertEqual(report["terminal_drop_filter"]["bestdori"]["removed"], 1)
        self.assertEqual(report["terminal_drop_filter"]["starfx"]["removed"], 1)
        self.assertEqual(report["terminal_drop_filter"]["hhwx"]["removed"], 0)

    def test_all_source_terminal_drops_preserve_live_high_for_pchip(self):
        def fake_fetch(_url):
            return {"cutoffs": [
                {"time": T0, "ep": 100},
                {"time": T0 + STEP, "ep": 140},
                {"time": T0 + 2 * STEP, "ep": 130},
            ]}

        end_at = T0 + 2 * STEP
        merged, report = fetch_and_merge_cutoffs(
            3, 999, 1000,
            start_at=T0,
            end_at=end_at,
            fetch_json=fake_fetch,
        )
        prepared = prepare_pchip_points(merged, T0, end_at)

        self.assertEqual(merged, [
            {"time": T0, "ep": 100},
            {"time": T0 + STEP, "ep": 140},
        ])
        self.assertEqual(prepared[-1], {"time": end_at, "ep": 140})
        self.assertEqual(
            sum(item["removed"] for item in report["terminal_drop_filter"].values()),
            3,
        )

    def test_source_urls_match_reference_project_routing(self):
        cn_urls = cutoff_source_urls(3, 321, 1000)

        self.assertEqual(set(cn_urls), {"bestdori", "starfx", "hhwx"})
        self.assertIn("/api/tracker/data?", cn_urls["bestdori"])
        self.assertIn("/api/tracker/data?", cn_urls["starfx"])
        self.assertIn("/api/bandori/tracker/data?", cn_urls["hhwx"])
        for url in cn_urls.values():
            self.assertIn("server=3&event=321&tier=1000", url)

        self.assertEqual(set(cutoff_source_urls(0, 321, 1000)), {"bestdori", "starfx"})
        self.assertEqual(set(cutoff_source_urls(1, 321, 1000)), {"bestdori"})

    def test_normalize_rejects_invalid_and_out_of_window_points(self):
        points, report = normalize_cutoff_points(
            [
                {"time": T0, "ep": "100"},
                {"timestamp": T0 + STEP, "value": 200},
                {"time": T0 + STEP, "ep": 210},
                {"time": T0 + 2 * STEP, "ep": -1},
                {"time": "bad", "ep": 300},
                {"time": T0 + 4 * STEP, "ep": 400},
            ],
            start_at=T0,
            end_at=T0 + 3 * STEP,
        )

        self.assertEqual(points, [
            {"time": T0, "ep": 100},
            {"time": T0 + STEP, "ep": 200},
        ])
        self.assertEqual(report["invalid"], 2)
        self.assertEqual(report["outside_event_window"], 1)
        self.assertEqual(report["duplicate_time"], 1)

    def test_starfx_removes_spike_and_trusted_envelope_outlier(self):
        trusted = [
            {"time": T0, "ep": 100_000},
            {"time": T0 + STEP, "ep": 120_000},
            {"time": T0 + 2 * STEP, "ep": 140_000},
            {"time": T0 + 3 * STEP, "ep": 160_000},
        ]
        starfx = [
            {"time": T0, "ep": 100_000},
            {"time": T0 + STEP // 2, "ep": 110_000},
            {"time": T0 + STEP, "ep": 999_999},  # 单点尖峰
            {"time": T0 + STEP + STEP // 2, "ep": 130_000},
            {"time": T0 + 2 * STEP + STEP // 2, "ep": 500_000},  # 单调但超出可信包络
        ]

        cleaned, report = filter_starfx_outliers(starfx, trusted)

        self.assertEqual([point["ep"] for point in cleaned], [100_000, 110_000, 130_000])
        self.assertEqual(report["removed"], 2)
        self.assertEqual(
            {point["reason"] for point in report["rejected"]},
            {"isolated_monotonic_violation", "trusted_source_envelope"},
        )

    def test_starfx_keeps_newer_point_when_trusted_sources_are_stale(self):
        trusted = [
            {"time": T0, "ep": 100_000},
            {"time": T0 + STEP, "ep": 120_000},
        ]
        starfx = [
            {"time": T0 + 2 * STEP, "ep": 160_000},
        ]

        cleaned, report = filter_starfx_outliers(starfx, trusted)

        self.assertEqual(cleaned, starfx)
        self.assertEqual(report["removed"], 0)

    def test_starfx_rejects_high_value_at_same_time_as_trusted_source(self):
        trusted = [{"time": T0, "ep": 100_000}]
        starfx = [{"time": T0, "ep": 900_000}]

        cleaned, report = filter_starfx_outliers(starfx, trusted)

        self.assertEqual(cleaned, [])
        self.assertEqual(report["removed"], 1)
        self.assertEqual(report["rejected"][0]["reason"], "trusted_source_envelope")

    def test_merge_takes_union_and_prefers_trusted_source_on_timestamp_conflict(self):
        merged, report = merge_cutoff_sources({
            "bestdori": [
                {"time": T0, "ep": 100},
                {"time": T0 + STEP, "ep": 200},
            ],
            "starfx": [
                {"time": T0 + STEP // 2, "ep": 150},
                {"time": T0 + STEP, "ep": 999},
            ],
            "hhwx": [
                {"time": T0 + 3 * STEP // 2, "ep": 250},
            ],
        })

        self.assertEqual(merged, [
            {"time": T0, "ep": 100},
            {"time": T0 + STEP // 2, "ep": 150},
            {"time": T0 + STEP, "ep": 200},
            {"time": T0 + 3 * STEP // 2, "ep": 250},
        ])
        self.assertEqual(report["same_time_conflicts"], 1)

    def test_fetch_uses_remaining_sources_when_one_source_fails(self):
        def fake_fetch(url):
            if "bestdori.com" in url:
                return {"cutoffs": [{"time": T0, "ep": 100}]}
            if "starfreedomx" in url:
                raise RuntimeError("temporary failure")
            if "hhwx.org" in url:
                return {"result": True, "cutoffs": [{"time": T0 + STEP, "ep": 200}]}
            raise AssertionError(url)

        merged, report = fetch_and_merge_cutoffs(
            3, 999, 1000,
            start_at=T0,
            end_at=T0 + 2 * STEP,
            fetch_json=fake_fetch,
        )

        self.assertEqual(merged, [
            {"time": T0, "ep": 100},
            {"time": T0 + STEP, "ep": 200},
        ])
        self.assertEqual(set(report["successful_sources"]), {"bestdori", "hhwx"})
        self.assertEqual(report["sources"]["starfx"]["status"], "error")


if __name__ == "__main__":
    unittest.main()
