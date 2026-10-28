import json
from http import HTTPStatus
from typing import Final

import pytest
from django.http import HttpResponse
from freezegun.api import FrozenDateTimeFactory
from inline_snapshot import snapshot

from dmr import Controller, modify
from dmr.plugins.pydantic import PydanticFastSerializer
from dmr.test import DMRAsyncRequestFactory, DMRRequestFactory
from dmr.throttling import AsyncThrottle, Rate, SyncThrottle
from dmr.throttling.algorithms import TokenBucket
from dmr.throttling.backends import SyncDjangoCache
from dmr.throttling.cache_keys import RemoteAddr
from dmr.throttling.headers import RateLimitIETFDraft

_ATTEMPTS: Final = 2
_RATE: Final = 10


class _SyncController(Controller[PydanticFastSerializer]):
    @modify(
        throttling=[
            SyncThrottle(
                _ATTEMPTS,
                _RATE,
                algorithm=TokenBucket(),
            ),
        ],
    )
    def get(self) -> str:
        return 'inside'


def test_token_bucket_sync_consume_and_reject(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Consume all tokens, then get rejected."""
    # Two requests consume all tokens:
    for _ in range(_ATTEMPTS):
        request = dmr_rf.get('/whatever/')
        response = _SyncController.as_view()(request)
        assert isinstance(response, HttpResponse)
        assert response.status_code == HTTPStatus.OK

    # Third is rejected:
    request = dmr_rf.get('/whatever/')
    response = _SyncController.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert response.headers == {
        'X-RateLimit-Limit': '2',
        'X-RateLimit-Remaining': '0',
        'X-RateLimit-Reset': str(_RATE // _ATTEMPTS),
        'Retry-After': str(_RATE // _ATTEMPTS),
        'Content-Type': 'application/json',
    }
    assert json.loads(response.content) == snapshot({
        'detail': [
            {'msg': 'Too many requests', 'type': 'ratelimit'},
        ],
    })


def test_token_bucket_smooth_refill(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Verify smooth refilling: partial time grants partial tokens."""
    # Consume all tokens:
    for _ in range(_ATTEMPTS):
        request = dmr_rf.get('/whatever/')
        response = _SyncController.as_view()(request)
        assert response.status_code == HTTPStatus.OK

    # Rejected while empty:
    request = dmr_rf.get('/whatever/')
    response = _SyncController.as_view()(request)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    # After several seconds one token is refilled:
    freezer.tick(delta=_RATE / _ATTEMPTS)

    request = dmr_rf.get('/whatever/')
    response = _SyncController.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK

    # No tokens left again - immediately rejected:
    request = dmr_rf.get('/whatever/')
    response = _SyncController.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS


def test_token_bucket_waits_for_full_token(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Reject requests until a whole token is refilled."""
    # Consume all tokens:
    for _ in range(_ATTEMPTS):
        request = dmr_rf.get('/whatever/')
        response = _SyncController.as_view()(request)
        assert response.status_code == HTTPStatus.OK

    # A partial refill is not enough for a whole request:
    freezer.tick(delta=1)

    request = dmr_rf.get('/whatever/')
    response = _SyncController.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    # This is the `<` vs `>=` boundary: an exactly full token
    # is allowed to be consumed.
    freezer.tick(delta=(_RATE // _ATTEMPTS) - 1)

    request = dmr_rf.get('/whatever/')
    response = _SyncController.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK


def test_token_bucket_full_refill(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """After full duration the bucket is full again."""
    # Consume all tokens:
    for _ in range(_ATTEMPTS):
        request = dmr_rf.get('/whatever/')
        response = _SyncController.as_view()(request)
        assert response.status_code == HTTPStatus.OK

    # After full duration all tokens are back:
    freezer.tick(delta=_RATE)

    for _ in range(_ATTEMPTS):
        request = dmr_rf.get('/whatever/')
        response = _SyncController.as_view()(request)
        assert isinstance(response, HttpResponse)
        assert response.status_code == HTTPStatus.OK


def test_token_bucket_idle_burst(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """An idle client can burst up to ``max_requests`` requests at once."""
    # Spend just one token:
    request = dmr_rf.get('/whatever/')
    response = _SyncController.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK

    # Tokens are refilled up to the full capacity during the idle period:
    freezer.tick(delta=_RATE // 2)

    # The idle client still can make all requests at once:
    for _ in range(_ATTEMPTS):
        request = dmr_rf.get('/whatever/')
        response = _SyncController.as_view()(request)
        assert isinstance(response, HttpResponse)
        assert response.status_code == HTTPStatus.OK

    # But, the next one is rejected, because the average
    # rate is still held:
    request = dmr_rf.get('/whatever/')
    response = _SyncController.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS


@pytest.mark.parametrize(
    'rate',
    [Rate.second, Rate.minute, Rate.hour, Rate.day],
)
def test_token_bucket_rates(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
    *,
    rate: Rate,
) -> None:
    """Rates work correctly with the token bucket."""

    class _Controller(Controller[PydanticFastSerializer]):
        throttling = [
            SyncThrottle(
                1,
                rate,
                algorithm=TokenBucket(),
                backend=SyncDjangoCache(allow_unsafe_cache=True),
            ),
        ]

        def get(self) -> str:
            return 'ok'

    # First is ok:
    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK

    # Second is rate limited:
    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    # After full duration, it is ok:
    freezer.tick(delta=int(rate))

    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK


class _TwoEndpointsController(Controller[PydanticFastSerializer]):
    @modify(
        throttling=[
            SyncThrottle(
                1,
                Rate.second,
                algorithm=TokenBucket(),
            ),
        ],
    )
    def get(self) -> str:
        return 'get'

    @modify(
        throttling=[
            SyncThrottle(
                1,
                Rate.second,
                algorithm=TokenBucket(),
            ),
        ],
    )
    def put(self) -> str:
        return 'put'


def test_token_bucket_per_endpoint_isolation(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Different endpoints have separate buckets."""
    # Consume the GET bucket:
    request = dmr_rf.get('/whatever/')
    response = _TwoEndpointsController.as_view()(request)
    assert response.status_code == HTTPStatus.OK

    # GET is now rejected:
    request = dmr_rf.get('/whatever/')
    response = _TwoEndpointsController.as_view()(request)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    # PUT still works (separate bucket):
    request = dmr_rf.put('/whatever/')
    response = _TwoEndpointsController.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK

    # Now it is also empty:
    request = dmr_rf.put('/whatever/')
    response = _TwoEndpointsController.as_view()(request)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS


_ODD_ATTEMPTS: Final = 3


class _AsyncController(Controller[PydanticFastSerializer]):
    throttling = [
        AsyncThrottle(
            _ODD_ATTEMPTS,
            _RATE,
            algorithm=TokenBucket(),
            response_headers=[RateLimitIETFDraft()],
        ),
        AsyncThrottle(
            30,  # noqa: WPS432
            Rate.minute,
            algorithm=TokenBucket(),
            response_headers=[RateLimitIETFDraft()],
            cache_key=RemoteAddr(name='min'),
        ),
    ]

    async def get(self) -> str:
        return 'ok'


@pytest.mark.asyncio
async def test_token_bucket_async(
    dmr_async_rf: DMRAsyncRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Async controllers work with the token bucket algorithm."""
    # Consume all tokens:
    for _ in range(_ODD_ATTEMPTS):
        request = dmr_async_rf.get('/whatever/')
        response = await dmr_async_rf.wrap(_AsyncController.as_view()(request))
        assert isinstance(response, HttpResponse)
        assert response.status_code == HTTPStatus.OK

    # Rejected:
    request = dmr_async_rf.get('/whatever/')
    response = await dmr_async_rf.wrap(_AsyncController.as_view()(request))
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert response.headers == {
        'RateLimit-Policy': '3;w=10;name="RemoteAddr", 30;w=60;name="min"',
        'RateLimit': '"RemoteAddr";r=0;t=4',
        'Content-Type': 'application/json',
    }

    # After a partial refill one more is allowed:
    freezer.tick(delta=4)  # `_ceil_div(10, 3)` result

    request = dmr_async_rf.get('/whatever/')
    response = await dmr_async_rf.wrap(_AsyncController.as_view()(request))
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK

    # After a full refill all tokens are back:
    freezer.tick(delta=_RATE)

    for _ in range(_ODD_ATTEMPTS):
        request = dmr_async_rf.get('/whatever/')
        response = await dmr_async_rf.wrap(_AsyncController.as_view()(request))
        assert isinstance(response, HttpResponse)
        assert response.status_code == HTTPStatus.OK

    # But, next are rejected:
    request = dmr_async_rf.get('/whatever/')
    response = await dmr_async_rf.wrap(_AsyncController.as_view()(request))
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert response.headers == {
        'RateLimit-Policy': '3;w=10;name="RemoteAddr", 30;w=60;name="min"',
        'RateLimit': '"RemoteAddr";r=0;t=4',
        'Content-Type': 'application/json',
    }
