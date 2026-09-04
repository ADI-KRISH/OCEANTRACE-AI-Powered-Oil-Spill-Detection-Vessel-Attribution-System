# Broad real-data validation — attribution vs. SkyTruth Cerulean

Complements `validation/real_cases.py` (one hand-picked, easy real incident)
with statistical breadth: 25 real Sentinel-1 slicks (US shelf,
2022-2023) whose #1 probable source Cerulean labels as a vessel, compared to
this module's own ranking on the same slick + real MarineCadastre AIS.

**Not ground truth.** Cerulean's own ranker also uses AIS. This measures
*agreement with an independent operational system*, which is the best real
check available without enforcement records.

Reproduce:
```
python -m validation.cerulean_benchmark --build --n-cases 25 --max-days 12
python -m validation.cerulean_benchmark --run
```

```
===== Cerulean benchmark summary =====
cases run                     : 25
cases with >=1 candidate      : 24
Cerulean #1 vessel seen by us : 62%
  agree@1                    : 27%
  Cerulean #1 in our top-3   : 27%
  median rank we give it     : 35.0
our #1 within Cerulean top-5  : 33%
```

## Cases

```
 slick_id       date region  n_candidates  n_screened cerulean_r1    our_r1          our_r1_name cerulean_r1_in_candidates  our_rank_of_cerulean_r1 agree_at_1 cerulean_r1_in_our_top3 our_r1_in_cerulean_topk
  3607398 2022-05-08   gulf           235      1024.0   246767000 367611250                REBEL                      True                    196.0      False                   False                   False
  3607389 2022-05-08   gulf           230       906.0   367653220 236111902          SANCO SWORD                      True                     45.0      False                   False                   False
  3607509 2022-12-10   gulf           164       588.0   367609520 636015112      GLOBETROTTER II                      True                    156.0      False                   False                   False
  3607511 2022-12-10   gulf           168       610.0   368087000 210328000      FULMAR EXPLORER                      True                     26.0      False                   False                    True
  3737112 2023-01-08   gulf            99       223.0   538009386 368240920         HOS MAVERICK                      True                     71.0      False                   False                   False
  3736676 2023-01-08   gulf           113       422.0   563104700 369108000         CADE CANDIES                     False                      NaN      False                   False                    True
  3736672 2023-01-08   gulf           105       546.0   215233000 538004174     DUBAI BRILLIANCE                      True                     31.0      False                   False                   False
  3712935 2023-04-02   gulf            83       274.0   565513000 366495000     OVERSEAS CHINOOK                     False                      NaN      False                   False                   False
  3754377 2023-04-02   gulf            95       331.0   538008625 367178330            AUGER TLP                      True                     79.0      False                   False                   False
  3754399 2023-04-02   gulf            86       264.0   565513000 367178330            AUGER TLP                     False                      NaN      False                   False                   False
  3148154 2023-06-01   west           661      1326.0   311000111 367110020               TARDIS                      True                      5.0      False                   False                   False
  3233299 2023-06-01   gulf            65       185.0   367640510 368584000        HARVEY SPIRIT                     False                      NaN      False                   False                    True
  3320948 2023-06-01   gulf            82       425.0   367640510 367785160           BRUTUS TLP                     False                      NaN      False                   False                   False
  3109191 2023-08-12   gulf            92       303.0   367691280 477722700 PACIFIC INEOS BELSTA                      True                     43.0      False                   False                   False
  3285076 2023-08-12   gulf            63       198.0   356811000 367655260      SHELIA BORDELON                     False                      NaN      False                   False                   False
  3109175 2023-08-12   gulf            74       186.0   357982000 357982000           MARIANNE-G                      True                      1.0       True                    True                    True
  3178179 2023-08-24   gulf            90       370.0   367643110 369204000          POTTER TIDE                     False                      NaN      False                   False                   False
  3142417 2023-08-24   gulf            85       224.0   366048300 366048300             HOLSTEIN                      True                      1.0       True                    True                    True
  3178206 2023-08-24   gulf           100       380.0   366697630 369204000          POTTER TIDE                     False                      NaN      False                   False                   False
  3185536 2023-09-17   gulf            64       472.0   538005462 367493760               TYRANT                      True                     35.0      False                   False                   False
  3142492 2023-09-17   east             2        49.0   368219910 563029900       TAIPEI TRIUMPH                     False                      NaN      False                   False                   False
  3082161 2023-09-29    NaN             0         NaN         NaN       NaN                  NaN                       NaN                      NaN        NaN                     NaN                     NaN
  3093203 2023-09-29   east             7        31.0   353775000 353775000          MSC BARBARA                      True                      1.0       True                    True                    True
  3253980 2023-11-06   east           617       812.0   368222940 368222940       LAZY LIGHTNING                      True                      1.0       True                    True                    True
  3139247 2023-11-06   east           244       589.0   443987689 367101390             HATTERAS                      True                     60.0      False                   False                    True
```

## Reading this

- **Coverage** below 100% is the dominant limiter, not the scoring model:
  Cerulean draws on global satellite AIS; MarineCadastre is US-only, so a
  vessel it names may simply not be in our feed.
- When the vessel *is* in our feed, `track_match` and the behavioural features
  (`gap_over_origin`, `slow_steaming`, `loiter_score`) are what should be
  pulling it up the ranking -- if agreement stays low even there, that points
  at the origin cloud (sigma/age window) rather than the scorer.
