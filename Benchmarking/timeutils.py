
import time
from datetime import datetime, timezone, tzinfo
try:
    from zoneinfo import ZoneInfo  # stdlib (Python 3.9+)
except Exception:
    ZoneInfo = None
from typing import Optional

class Time:


    """
    Global static time utilities. Configure once via TimeUtils.configure(...)
    and call methods statically: TimeUtils.now(), TimeUtils.to_epoch(dt), ...
    """
    # Defaults (class-level; act as global config)
    #default: local
    # if configured: the configured
    # if overriden in function - the function's 
    _tz = None
    _fmt = None
    #_epoch_precision: _EpochPrecision = "s"  # "s","ms","us","ns"
    #_use_aware: bool = True  # return timezone-aware datetimes by default
    # ZoneInfo(_tz_name) if ZoneInfo else


    @classmethod
    def default(cls, *, tz: Optional[int] = None, fmt: Optional[int] = None):
        if tz:
            cls._tz = datetime.now().astimezone().tzinfo

        if fmt:
            cls._fmt = "%Y%m%d-%H%M%S"


    # ---------------- Configuration ----------------
    @classmethod
    def configure(cls, tz: Optional[str] = None, fmt: Optional[str] = None):
                  #epoch_precision: Optional[_EpochPrecision] = None, use_aware: Optional[bool] = None):
        """Set global defaults. Call at program start if you want non-default behavior."""
        if tz is not None:
            cls._tz = cls.timezone(tz)


        if fmt is not None:
            cls._fmt = cls.format(fmt)

        # if tz is None:
        #     cls.default(tz=1)
        # else:
        #     cls._tz = tz
        #     if ZoneInfo:
        #         cls._tz = ZoneInfo(tz_name)
        #     else:
        #         cls._tz = timezone.utc  # fallback; encourage installing zoneinfo


        # if epoch_precision is not None:
        #     if epoch_precision not in ("s", "ms", "us", "ns"):
        #         raise ValueError("epoch_precision must be one of 's','ms','us','ns'")
        #     cls._epoch_precision = epoch_precision
        # if use_aware is not None:
        #     cls._use_aware = bool(use_aware)

    @classmethod
    def tz_name(cls, tzname: str):
        if tzname.upper() == "UTC":
            target_tz = timezone.utc
        else:
            if ZoneInfo is None:
                raise RuntimeError("zoneinfo not available; pass tz as tzinfo or use 'UTC'")
            target_tz = ZoneInfo(tzname)
        return target_tz
    

    @classmethod
    def timezone(cls, tz):
        # resolve target timezone
        if tz is None:
            #the configured
            if cls._tz is None:
                cls.default(tz=1)
            
            target_tz = cls._tz
            
        elif isinstance(tz, str):
            # from tz name
            target_tz = cls.tz_name(tz)

        elif isinstance(tz, tzinfo):
            target_tz = tz
        else:
            raise TypeError("tz must be None, a timezone string or tzinfo")
        return target_tz


    @classmethod
    def format(cls, fmt):
        if fmt is None:
            if cls._fmt is None:
                cls.default(fmt=1)
            target_fmt = cls._fmt
            # dt = datetime.now()
            # s = dt.isoformat(timespec=cls.timespec())
            # # use "Z" for UTC to match RFC3339 canonical form
            # if dt.utcoffset() == timedelta(0):
            #     s = s.replace("+00:00", "Z")
        elif isinstance(fmt, str):
            #TODO:validate fmt is legitimate    
            target_fmt = fmt
            
        return target_fmt


    @classmethod
    def timespec(cls, spec = 's'):
        return "seconds" #spec


    @classmethod
    def now(cls)-> float:
        # returns unix epoch at UTC 
        return time.time()


    @classmethod
    def timestamp(cls, epoch:float = None, tz = None, fmt: str = None, timespec: str = "microseconds") -> str:
        """
        Return current time as RFC3339-with-offset (default) or formatted string.
        - fmt: optional strftime format. If None returns ISO8601/RFC3339 with offset (microseconds by default).
        - tz: None -> local timezone. If str, interpreted as IANA zone name (or "UTC"). If tzinfo provided it's used.
        - timespec: passed to datetime.isoformat (e.g. "auto", "seconds", "milliseconds", "microseconds").
        The actual instant is captured immediately on call.
        """
        # capture instant immediately
        if epoch is None:
            epoch = cls.now()

        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)

        target_tz = cls.timezone(tz)

        fmt = cls.format(fmt)

        dt = dt.astimezone(target_tz)

        return dt.strftime(fmt)


    @classmethod
    def sleep(cls, interval):
            time.sleep(interval)
