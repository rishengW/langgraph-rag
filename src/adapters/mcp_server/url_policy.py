"""HTTPS and destination policy for MCP-supplied and discovered source URLs."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Sequence
from urllib.parse import urlsplit

AsyncResolver = Callable[[str, int], Awaitable[Sequence[str]]]
SyncResolver = Callable[[str, int], Sequence[str]]

_METADATA_HOSTS = frozenset(
    {
        "metadata",
        "metadata.google.internal",
        "metadata.azure.internal",
        "instance-data.ec2.internal",
    }
)


class UnsafeSourceURLError(ValueError):
    """A source URL violates the inbound MCP destination policy."""


def _default_sync_resolver(host: str, port: int) -> Sequence[str]:
    records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


async def _default_async_resolver(host: str, port: int) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


def _parsed_destination(url: str) -> tuple[str, int]:
    value = url.strip()
    if not value or len(value) > 4096:
        raise UnsafeSourceURLError("Source URL is empty or too long")
    try:
        parsed = urlsplit(value)
        port = parsed.port or 443
    except ValueError:
        raise UnsafeSourceURLError("Source URL is malformed") from None
    if parsed.scheme.lower() != "https":
        raise UnsafeSourceURLError("Only HTTPS source URLs are accepted")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeSourceURLError("Source URLs cannot contain credentials")
    if not parsed.hostname:
        raise UnsafeSourceURLError("Source URL must contain a valid host")
    if port != 443:
        raise UnsafeSourceURLError("Source URL ports other than 443 are not accepted")
    try:
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise UnsafeSourceURLError("Source URL host is malformed") from None
    if not host or host in _METADATA_HOSTS or host.endswith(".internal") or host.endswith(".local"):
        raise UnsafeSourceURLError("Source URL host is prohibited")
    try:
        literal_address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        pass
    else:
        _validate_address(str(literal_address))
    return host, port


def _validate_address(raw_address: str) -> None:
    try:
        address = ipaddress.ip_address(raw_address.split("%", 1)[0])
    except ValueError:
        raise UnsafeSourceURLError("Source host resolved to a malformed address") from None
    if (
        not address.is_global
        or address.is_loopback
        or address.is_link_local
        or address.is_private
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    ):
        raise UnsafeSourceURLError("Source host resolved to a prohibited address")


def _validate_resolved(addresses: Sequence[str]) -> None:
    if not addresses:
        raise UnsafeSourceURLError("Source host did not resolve")
    for address in addresses:
        _validate_address(address)


class URLValidator:
    """Resolve every host and reject any unsafe address before application I/O.

    This validator is deliberately injectable for deterministic tests. It performs
    pre-invocation DNS validation, but the legacy fetcher does not support binding
    the validated address to the later connection; that remaining rebinding window
    is documented and must not be represented as connection-boundary pinning.
    """

    def __init__(
        self,
        *,
        async_resolver: AsyncResolver = _default_async_resolver,
        sync_resolver: SyncResolver = _default_sync_resolver,
    ) -> None:
        self._async_resolver = async_resolver
        self._sync_resolver = sync_resolver

    async def validate(self, urls: Sequence[str], *, maximum: int) -> list[str]:
        values = _bounded_values(urls, maximum)
        validated: list[str] = []
        for value in values:
            host, port = _parsed_destination(value)
            try:
                addresses = await self._async_resolver(host, port)
            except (OSError, UnicodeError):
                raise UnsafeSourceURLError("Source host could not be resolved safely") from None
            _validate_resolved(addresses)
            validated.append(value)
        return validated

    def validate_sync(self, urls: Sequence[str], *, maximum: int) -> list[str]:
        values = _bounded_values(urls, maximum)
        validated: list[str] = []
        for value in values:
            host, port = _parsed_destination(value)
            try:
                addresses = self._sync_resolver(host, port)
            except (OSError, UnicodeError):
                raise UnsafeSourceURLError("Source host could not be resolved safely") from None
            _validate_resolved(addresses)
            validated.append(value)
        return validated


def _bounded_values(urls: Sequence[str], maximum: int) -> list[str]:
    if len(urls) > maximum:
        raise UnsafeSourceURLError(f"At most {maximum} source URLs are accepted")
    values = [str(url).strip() for url in urls]
    if len(set(values)) != len(values):
        raise UnsafeSourceURLError("Duplicate source URLs are not accepted")
    return values


__all__ = ["AsyncResolver", "SyncResolver", "URLValidator", "UnsafeSourceURLError"]
