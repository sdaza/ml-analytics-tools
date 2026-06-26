# Changelog

## [0.7.1](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.7.0...ml-analytics-tools-v0.7.1) (2026-06-26)


### Bug Fixes

* refactor SFConnector to delegate table management to SparkTableManager and add spark_connector module ([73df3b7](https://github.com/sdaza/ml-analytics-tools/commit/73df3b71e3666fba5fb452d6b849b8a682e1fddb))


### Documentation

* add SparkTableManager documentation and usage examples ([e5205b3](https://github.com/sdaza/ml-analytics-tools/commit/e5205b310c821d2918cc9d28d9cc4c3a23e28cf1))

## [0.7.0](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.6.0...ml-analytics-tools-v0.7.0) (2026-06-25)


### Features

* **gsheet:** resolve credentials from GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_SERVICE_ACCOUNT_PATH ([e179fca](https://github.com/sdaza/ml-analytics-tools/commit/e179fca30553cd548be249ecc10f00c00b8d8dda))

## [0.6.0](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.5.5...ml-analytics-tools-v0.6.0) (2026-06-25)


### Features

* make modeling deps optional and add local Databricks Connect support ([a1591c3](https://github.com/sdaza/ml-analytics-tools/commit/a1591c3fafcb51b8e9f3f5663f20e077cc32aca6))


### Bug Fixes

* **ci:** install [modeling] extra and skip modeling tests when absent ([77d7d3e](https://github.com/sdaza/ml-analytics-tools/commit/77d7d3e7bc1af299765596e2d95a0a3dfe2fe632))

## [0.5.5](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.5.4...ml-analytics-tools-v0.5.5) (2026-06-24)


### Bug Fixes

* clear sheet before writing ([c213751](https://github.com/sdaza/ml-analytics-tools/commit/c2137515d9319e12d41c5625e5295a69d5135944))

## [0.5.4](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.5.3...ml-analytics-tools-v0.5.4) (2026-06-24)


### Bug Fixes

* update key-pair setup for snowflake ([aeb75c9](https://github.com/sdaza/ml-analytics-tools/commit/aeb75c91baefc8fd060d72265ba2c471814216da))

## [0.5.3](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.5.2...ml-analytics-tools-v0.5.3) (2026-06-23)


### Bug Fixes

* enhance SQL query handling to support comment-aware placeholder substitution ([d7d2c68](https://github.com/sdaza/ml-analytics-tools/commit/d7d2c68cb08538d4439b41a04cbeb62cdf63075d))

## [0.5.2](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.5.1...ml-analytics-tools-v0.5.2) (2026-06-23)


### Bug Fixes

* **data_connector, sf_connector:** enhance SQL query handling to skip formatting for inline comments ([020d7ce](https://github.com/sdaza/ml-analytics-tools/commit/020d7cec80cea349cbafe12e08e3d2185afedb3c))

## [0.5.1](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.5.0...ml-analytics-tools-v0.5.1) (2026-06-19)


### Bug Fixes

* **sf_connector:** enhance _resolve_query to format inline SQL queries with kwargs ([61d7313](https://github.com/sdaza/ml-analytics-tools/commit/61d7313cc614dffe74bd0710d0ef75fe6f41014f))

## [0.5.0](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.4.7...ml-analytics-tools-v0.5.0) (2026-06-19)


### Features

* **sf_connector:** add options to drop existing tables and overwrite schema in save_to_uc ([e564bae](https://github.com/sdaza/ml-analytics-tools/commit/e564bae9ed2a7539a741711ed3387aa2ca32bf80))

## [0.4.7](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.4.6...ml-analytics-tools-v0.4.7) (2026-06-19)


### Bug Fixes

* **sf_connector:** add query wrapping to preserve comments for Snowflake connector ([489400f](https://github.com/sdaza/ml-analytics-tools/commit/489400ff1315e4bac71ba3685c39bd39b1f45d44))

## [0.4.6](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.4.5...ml-analytics-tools-v0.4.6) (2026-06-18)


### Bug Fixes

* **model_manager, sf_connector, utils:** enhance model management and SQL handling ([52f5763](https://github.com/sdaza/ml-analytics-tools/commit/52f576373aadd21bfe2e5fad1dfe300544130106))

## [0.4.5](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.4.4...ml-analytics-tools-v0.4.5) (2026-06-18)


### Bug Fixes

* **data_connector, sf_connector:** add methods to lowercase DataFrame columns and handle decimal types for pandas conversion ([09cf1a9](https://github.com/sdaza/ml-analytics-tools/commit/09cf1a94f4da7d343d9d4b8c1fad16a78dc8e6e9))

## [0.4.4](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.4.3...ml-analytics-tools-v0.4.4) (2026-06-17)


### Bug Fixes

* **sf_connector:** add save_pipeline_to_uc for YAML-ordered SQL execution and Unity Catalog saving ([a2b91b1](https://github.com/sdaza/ml-analytics-tools/commit/a2b91b1dc316b9169299f0950cee5489e72ac36e))
* **utils:** implement resolve_sql_query_paths to normalize SQL query paths ([a2b91b1](https://github.com/sdaza/ml-analytics-tools/commit/a2b91b1dc316b9169299f0950cee5489e72ac36e))


### Documentation

* update README and SF_CONNECTOR_USAGE with new save_pipeline_to_uc usage examples ([a2b91b1](https://github.com/sdaza/ml-analytics-tools/commit/a2b91b1dc316b9169299f0950cee5489e72ac36e))

## [0.4.3](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.4.2...ml-analytics-tools-v0.4.3) (2026-06-16)


### Bug Fixes

* **utils:** enhance load_sql_query to support fallback paths for SQL files ([bdc72d8](https://github.com/sdaza/ml-analytics-tools/commit/bdc72d80758fdeac07fd8ce5f4cd46d8a82ace26))

## [0.4.2](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.4.1...ml-analytics-tools-v0.4.2) (2026-06-16)


### Bug Fixes

* **docs:** update SFConnector documentation for Unity Catalog support and SQL file queries ([8d6514c](https://github.com/sdaza/ml-analytics-tools/commit/8d6514c90d45fc44551eaa599b63917a33ca9b2e))
* **sf_connector:** enhance SFConnector with query resolution and Unity Catalog support ([6ee4c9b](https://github.com/sdaza/ml-analytics-tools/commit/6ee4c9bfdb9c611a565193ded0bc982c5b50dcdf))

## [0.4.1](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.4.0...ml-analytics-tools-v0.4.1) (2026-06-16)


### Bug Fixes

* **docs:** update GCP project example in README and tests for clarity ([135779f](https://github.com/sdaza/ml-analytics-tools/commit/135779f8fd6adaaa2652310e290537f38fc18ebc))
* **docs:** update GCP project example in README and tests for clarity ([a90a9de](https://github.com/sdaza/ml-analytics-tools/commit/a90a9deabfa6b58fba7754997265ed1971675a8f))

## [0.4.0](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.3.0...ml-analytics-tools-v0.4.0) (2026-06-16)


### Features

* **SFConnector:** add Spark-based Snowflake connector for reading/writing DataFrames ([84aaf97](https://github.com/sdaza/ml-analytics-tools/commit/84aaf97306841129a33b36c619fb545401f45cfe))
* **SFConnector:** infer secret scope from Databricks user for seamless usage ([8d06f41](https://github.com/sdaza/ml-analytics-tools/commit/8d06f411604eab3c99359aae4a9e11165deaeea5))
* **utils:** add lazy loading for Databricks dbutils and enhance credential retrieval ([4405092](https://github.com/sdaza/ml-analytics-tools/commit/4405092bc4c7c48bbfc1c7d1358cb670c9b803c7))

## [0.3.0](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.2.1...ml-analytics-tools-v0.3.0) (2026-06-11)


### Features

* add Snowflake support to DataConnector and update README with configuration details ([7237355](https://github.com/sdaza/ml-analytics-tools/commit/7237355a18dd9b91acabe249a080628cfabfed7d))
* add Snowflake support to DataConnector and update README with configuration details ([a542ae2](https://github.com/sdaza/ml-analytics-tools/commit/a542ae2d8acc55c1a6b43d5cadde0f384c742455))
* **gsheet:** add OAuth installed-app flow (no-token path) ([7d51bb4](https://github.com/sdaza/ml-analytics-tools/commit/7d51bb40b8b5a7cee0723ad3bb8f438ffadb63e4))
* **gsheet:** document OAuth env vars and service_account_email None behavior ([7d2bfb4](https://github.com/sdaza/ml-analytics-tools/commit/7d2bfb43a0e678d749b2136f188e2d04a65c8f9b))


### Bug Fixes

* **gitignore:** include docs/superpowers directory in .gitignore ([8326c80](https://github.com/sdaza/ml-analytics-tools/commit/8326c80649894079efde6f3600e9f19248b29ffa))
* **gsheet:** tolerate corrupt OAuth token, fix return type, harden test fixture ([841908a](https://github.com/sdaza/ml-analytics-tools/commit/841908afe049207b8f52f85557afa1770b46adec))


### Documentation

* **README:** add note on SSO token caching in OS keychain ([0eef0b3](https://github.com/sdaza/ml-analytics-tools/commit/0eef0b3c05e72a3c15ec5bf5cd2ad46514779bbf))

## [0.2.1](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.2.0...ml-analytics-tools-v0.2.1) (2026-05-20)


### Bug Fixes

* add badges for CI, GitHub release, Python version, and license in README ([2486a13](https://github.com/sdaza/ml-analytics-tools/commit/2486a13cf51010deebdd4842d803e1d6d2bfe615))
* add PyPI badge to README for package version visibility ([23e0f84](https://github.com/sdaza/ml-analytics-tools/commit/23e0f8407486fd5779e303e504f1e026d7cfcc0e))

## [0.2.0](https://github.com/sdaza/ml-analytics-tools/compare/ml-analytics-tools-v0.1.0...ml-analytics-tools-v0.2.0) (2026-05-20)


### Features

* update release workflow and versioning for package release ([f1bc6af](https://github.com/sdaza/ml-analytics-tools/commit/f1bc6af7e728e2e227d1981fcbe5dafde71f4c8f))


### Bug Fixes

* improve documentation ([a67f72c](https://github.com/sdaza/ml-analytics-tools/commit/a67f72cdcb73a6ef4e5bff39dbd1000962539e9c))

## 0.1.0 (2026-05-20)


### Bug Fixes

* improve documentation ([a67f72c](https://github.com/sdaza/ml-analytics-tools/commit/a67f72cdcb73a6ef4e5bff39dbd1000962539e9c))
