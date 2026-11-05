# Regulus 21 - API reference

**Generated from the route definitions on 2026-09-22.** 108 endpoints.

Every row was read out of `regulus-backend/src/app/**/route.ts` and the handler
it mounts, so this states what the code does rather than what it was once
intended to do. It is generated - see the last section before editing it.

---

## How to read this

**Base URL.** The backend runs at `http://localhost:3001` in development. The
frontend reaches it through `NEXT_PUBLIC_API_BASE_URL`, which defaults to
`/api` on the same origin. Production and sandbox URLs are not deployed yet.

**Auth** is one of five things, and they are not interchangeable:

| | |
|---|---|
| `session` | The `regulus_session` cookie, HttpOnly and SameSite=Lax. Obtained from `POST /api/auth/login`. |
| `public` | No credential. |
| `API key` | `Authorization: Bearer rk_live_…`. Partner integrations only - a browser never holds one. |
| `Stripe signature` | The `stripe-signature` header, verified against the signing secret. |
| `signed URL` | A key signed by this application, carried in the path. |

**Capability** is the permission the signed-in role must hold. The matrix is in
`regulus-backend/src/lib/auth/rbac.ts` and is enforced on the server; the
frontend keeps a copy only to decide what to render.

| Role | Holds |
|---|---|
| `END` | query.ask, corpus.browse, answer.export, library.manage, billing.manageOwn |
| `ORG` | the above, plus org.manageUsers |
| `REV` | query.ask, corpus.browse, library.manage, content.author, content.approve |
| `SYS` | query.ask, corpus.browse, answer.export, library.manage, admin.accounts, admin.settings, admin.dashboards, admin.videos, admin.audit, admin.gateTester, admin.harness |

A dash in the Capability column means the endpoint needs a session but no
particular permission - not that it is open.

**Idempotent** marks an endpoint that honours `X-Idempotency-Key`: a repeat with
the same key returns the first result instead of running again. It appears where
money, credits or a billed answer are involved.

**Errors** share one shape:

```json
{ "code": "validation_error", "message": "…", "fieldErrors": { "tierId": "…" } }
```

| Status | When |
|---|---|
| 400 / 415 | Malformed body, or the wrong content type |
| 401 | No session, or an expired one |
| 403 | Signed in, but the role lacks the capability - or the plan does not reach it |
| 404 | Not found, and deliberately not distinguishable from "not yours" |
| 409 | A conflict, such as subscribing twice |
| 413 | Body over the limit |
| 422 | Schema failure, and out-of-scope questions |
| 429 | Rate limited, with `Retry-After` in seconds |
| 500 / 503 | An unexpected failure, or a dependency that is not configured |

---

## Partner API (v1)

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `POST` | `/v1/chat` | API key | - | Veridian's contract, wrapped in their response envelope. Body limit 64 KB. |
| `POST` | `/v1/queries` | API key | - | idempotent; body: `partnerQuerySchema` |

## Authentication

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `POST` | `/api/auth/forgot-password` | public | - | body: `forgotPasswordSchema`; audited as `auth.forgot_password` |
| `POST` | `/api/auth/login` | public | - | body: `loginSchema`; audited as `auth.login` |
| `POST` | `/api/auth/logout` | session | - | audited as `auth.logout` |
| `POST` | `/api/auth/mfa` | public | - | body: `mfaVerifySchema`; audited as `auth.mfa_verify` |
| `POST` | `/api/auth/register` | public | - | idempotent; body: `registerSchema`; audited as `auth.register` |
| `POST` | `/api/auth/reset-password` | public | - | body: `resetPasswordSchema`; audited as `auth.reset_password` |
| `GET` | `/api/auth/session` | session | - | - |
| `POST` | `/api/auth/switch-role` | session | - | body: `switchRoleSchema`; audited as `auth.switch_role` |

## Account, profile and MFA

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `POST` | `/api/account/avatar` | session | - | body: `updateAvatarSchema`; audited as `account.avatar_update` |
| `GET` | `/api/account/avatar/:userId` | session | - | - |
| `GET` | `/api/account/credits` | session | - | - |
| `GET` | `/api/account/credits/history` | session | - | - |
| `POST` | `/api/account/credits/top-up` | session | `billing.manageOwn` | idempotent; body: `topUpSchema`; audited as `billing.top_up_checkout` |
| `GET` | `/api/account/mfa` | session | - | - |
| `POST` | `/api/account/mfa` | session | - | body: `updateMfaSchema`; audited as `account.mfa_update` |
| `POST` | `/api/account/password` | session | - | body: `changePasswordSchema`; audited as `account.password_change` |
| `PATCH` | `/api/account/profile` | session | - | body: `updateProfileSchema`; audited as `account.profile_update` |

## Asking a question

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `POST` | `/api/query` | session | - | body: `askSchema`; audited as `answer.query` |

## Answers

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `POST` | `/api/answers/:answerId/guidance` | session | `query.ask` | - |
| `POST` | `/api/answers/export` | session | `answer.export` | body: `exportSchema`; audited as `answer.export` |

## Browsing Title 21

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/browse` | session | - | - |
| `GET` | `/api/browse/:subchapter` | session | - | - |
| `GET` | `/api/browse/:subchapter/:part` | session | - | - |

## Records

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/records` | session | - | query: `searchQuery` |
| `GET` | `/api/records/:id` | session | - | - |
| `GET` | `/api/records/:id/versions` | session | - | - |
| `GET` | `/api/records/lookup` | session | - | query: `lookupQuery` |

## Saved queries, bookmarks and history

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/library/bookmarks` | session | `library.manage` | - |
| `POST` | `/api/library/bookmarks` | session | `library.manage` | body: `bookmarkSchema` |
| `DELETE` | `/api/library/bookmarks/:id` | session | `library.manage` | - |
| `DELETE` | `/api/library/history` | session | `library.manage` | - |
| `GET` | `/api/library/history` | session | `library.manage` | query: `historyQuerySchema` |
| `GET` | `/api/library/history/:id` | session | `library.manage` | - |
| `GET` | `/api/library/saved` | session | `library.manage` | - |
| `POST` | `/api/library/saved` | session | `library.manage` | body: `saveSchema` |
| `DELETE` | `/api/library/saved/:id` | session | `library.manage` | - |

## Reviewed content

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/content/:recordId` | session | `content.author` | - |
| `PUT` | `/api/content/:recordId` | session | `content.author` | body: `draftSchema`; audited as `content.draft.save` |
| `POST` | `/api/content/:recordId/approve` | session | `content.approve` | body: `approveSchema`; audited as `content.approve` |
| `GET` | `/api/content/coverage` | session | `content.author or admin.dashboards` | Either capability admits: the Reviewer's own report and the operations view of it are the same figures. Checked inside the handler, because the wrapper takes a single capability. |
| `GET` | `/api/content/queue` | session | `content.author` | query: `queueQuerySchema` |

## Video

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `POST` | `/api/videos/:id/playback` | session | - | - |

## Media

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/media/:key` | signed URL | - | Serves local storage objects from a signed key. Answers 404 when a cloud storage provider is configured, because that provider issues URLs to its own domain. |

## Subscriptions and billing

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `POST` | `/api/billing/cancel` | session | `billing.manageOwn` | idempotent; body: `cancelSubscriptionSchema`; audited as `billing.cancel` |
| `POST` | `/api/billing/checkout` | session | `billing.manageOwn` | idempotent; audited as `billing.checkout_opened` |
| `GET` | `/api/billing/credit-packages` | session | - | - |
| `PATCH` | `/api/billing/plan` | session | `billing.manageOwn` | body: `changePlanSchema`; audited as `billing.plan_change` |
| `POST` | `/api/billing/portal-session` | session | `billing.manageOwn` | - |
| `POST` | `/api/billing/reconcile` | session | `billing.manageOwn` | idempotent; body: `reconcileSchema`; audited as `billing.checkout_reconcile` |
| `POST` | `/api/billing/subscribe` | session | `billing.manageOwn` | idempotent; body: `subscribeSchema`; audited as `billing.subscribe` |
| `GET` | `/api/billing/subscription` | session | `billing.manageOwn` | - |
| `POST` | `/api/billing/webhook` | Stripe signature | - | Stripe only. Authenticated by the `stripe-signature` header against the signing secret - no session, no key. An unverifiable event is refused with 400 so Stripe does not retry it. |

## Plan catalogue

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/tiers` | public | - | - |

## Organisations and seats

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/orgs/seats` | session | `org.manageUsers` | - |
| `PATCH` | `/api/orgs/seats` | session | `org.manageUsers` | body: `seatCountSchema`; audited as `org.seats_changed` |
| `DELETE` | `/api/orgs/seats/:id` | session | `org.manageUsers` | audited as `org.seat_released` |
| `PATCH` | `/api/orgs/seats/:id` | session | `org.manageUsers` | body: `updateMemberSchema`; audited as `org.member_updated` |
| `POST` | `/api/orgs/seats/invite` | session | `org.manageUsers` | idempotent; body: `inviteSeatSchema`; audited as `org.seat_invited` |

## Invitations

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/invitations/:token` | public | - | - |
| `POST` | `/api/invitations/:token/accept` | public | - | idempotent; body: `acceptInvitationSchema`; audited as `org.invitation_accepted` |

## Administration

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/admin/accounts` | session | `admin.accounts` | query: `accountsQuerySchema` |
| `GET` | `/api/admin/accounts/:id` | session | `admin.accounts` | - |
| `GET` | `/api/admin/action-log` | session | `admin.audit` | query: `actionLogQuerySchema` |
| `GET` | `/api/admin/ai-strategy` | session | `admin.settings` | - |
| `PUT` | `/api/admin/ai-strategy` | session | `admin.settings` | body: `selectStrategySchema`; audited as `admin.ai_strategy_change` |
| `GET` | `/api/admin/api-clients` | session | `admin.settings` | - |
| `POST` | `/api/admin/api-clients` | session | `admin.settings` | idempotent; body: `createApiClientSchema`; audited as `admin.api_client_create` |
| `PATCH` | `/api/admin/api-clients/:id` | session | `admin.settings` | idempotent; body: `updateApiClientSchema`; audited as `admin.api_client_update` |
| `POST` | `/api/admin/api-clients/:id/revoke` | session | `admin.settings` | idempotent; body: `revokeApiClientSchema`; audited as `admin.api_client_revoke` |
| `POST` | `/api/admin/api-clients/:id/rotate` | session | `admin.settings` | idempotent; body: `rotateApiKeySchema`; audited as `admin.api_key_rotate` |
| `POST` | `/api/admin/api-keys/:id/revoke` | session | `admin.settings` | idempotent; audited as `admin.api_key_revoke` |
| `GET` | `/api/admin/audit` | session | `admin.audit` | query: `auditQuerySchema` |
| `GET` | `/api/admin/audit/:id` | session | `admin.audit` | - |
| `POST` | `/api/admin/audit/export` | session | `admin.audit` | audited as `audit.export` |
| `GET` | `/api/admin/changes` | session | `admin.dashboards` | query: `changesQuerySchema` |
| `GET` | `/api/admin/changes/:id` | session | `admin.dashboards` | - |
| `GET` | `/api/admin/completeness` | session | `admin.dashboards` | - |
| `GET` | `/api/admin/credit-packages` | session | `admin.settings` | - |
| `PUT` | `/api/admin/credit-packages/:id` | session | `admin.settings` | idempotent; body: `creditPackagePatchSchema`; audited as `admin.credit_pack_update` |
| `GET` | `/api/admin/dashboard` | session | `admin.dashboards` | - |
| `GET` | `/api/admin/gate-failures` | session | `admin.audit` | query: `gateFailuresQuerySchema` |
| `GET` | `/api/admin/gate-failures/:id` | session | `admin.audit` | - |
| `GET` | `/api/admin/harness` | session | `admin.harness` | - |
| `POST` | `/api/admin/harness` | session | `admin.harness` | body: `startSchema`; audited as `harness.run.start` |
| `GET` | `/api/admin/harness/:id` | session | `admin.harness` | - |
| `GET` | `/api/admin/harness/questions` | session | `admin.harness` | - |
| `POST` | `/api/admin/harness/questions` | session | `admin.harness` | body: `importSchema`; audited as `harness.questions.import` |
| `GET` | `/api/admin/iso-crosswalk` | session | `admin.settings` | - |
| `POST` | `/api/admin/iso-crosswalk/:id/approve` | session | `admin.settings` | idempotent; audited as `admin.iso_crosswalk_approve` |
| `GET` | `/api/admin/iso-index/:id` | session | `admin.settings` | - |
| `POST` | `/api/admin/iso-index/:id/activate` | session | `admin.settings` | idempotent; audited as `admin.iso_index_activate` |
| `POST` | `/api/admin/iso-index/:id/approve` | session | `admin.settings` | idempotent; body: `approveDeliverySchema`; audited as `admin.iso_index_approve` |
| `POST` | `/api/admin/reviewers/invite` | session | `admin.accounts` | idempotent; body: `inviteReviewerSchema`; audited as `platform.reviewer_invited` |
| `GET` | `/api/admin/tiers` | session | `admin.settings` | - |
| `PUT` | `/api/admin/tiers/:id` | session | `admin.settings` | idempotent; body: `tierPatchSchema`; audited as `admin.tier_update` |
| `GET` | `/api/admin/videos` | session | `admin.videos` | query: `videoQuerySchema` |
| `POST` | `/api/admin/videos` | session | `admin.videos` | idempotent; body: `createVideoSchema`; audited as `video.created` |
| `PATCH` | `/api/admin/videos/:id` | session | `admin.videos` | body: `updateVideoSchema`; audited as `video.updated` |
| `POST` | `/api/admin/videos/:id/file` | session | `admin.videos` | Multipart upload, written directly rather than through the JSON route wrapper. Session and capability are checked inside the handler. |

## Verification

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `POST` | `/api/verify` | session | `admin.gateTester` | body: `gateTestSchema`; audited as `gate.test` |

## Contact

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `POST` | `/api/contact/enquiry` | public | - | idempotent; body: `enquirySchema`; audited as `contact.enquiry` |

## Internal

| Method | Path | Auth | Capability | Notes |
|---|---|---|---|---|
| `GET` | `/api/internal/health` | public | - | - |
| `GET` | `/api/internal/ready` | public | - | - |
| `GET` | `/api/internal/selftest` | public | - | - |
| `POST` | `/api/internal/selftest` | public | - | idempotent; audited as `selftest.post` |
| `GET` | `/api/internal/selftest/ratelimit` | public | - | - |

---

## What this table does not carry

- **The partner contract in full.** `POST /v1/chat` and `POST /v1/queries` -
  request shapes, all eight tasks, the answer states, real captured responses,
  rate limits and retry semantics - are in
  `docs/partner-api/Regulus-AI-Veridian-API-Handoff.md`.
- **Answer states.** `VERIFIED`, `PARTIAL`, `NO_SOURCE`, `NOT_COVERED`,
  `NONEXISTENT_SECTION`, `SECTION_REMOVED`, `ISO_POINTER`,
  `ISO_TEXT_WITHHELD`, `ENTITLEMENT_BLOCKED`, `GATE_FAILURE`,
  `CORPUS_LISTING`, `OUT_OF_SCOPE`. Explained in the same document, and they
  apply to the internal query endpoints too.
- **Flows that cross several endpoints.** Paying for a plan change, for
  instance, runs from `PATCH /api/billing/plan` through a processor-hosted page
  to `POST /api/billing/webhook`, and no single row describes it.

## Regenerating

Generated - rerun rather than editing, so it cannot drift from the routes:

```
node scripts/api-reference.mjs
```
