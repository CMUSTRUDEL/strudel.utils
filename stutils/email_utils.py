
import os

import pandas as pd
import six

from typing import Optional, Tuple, Union

from stutils.decorators import memoize


class InvalidEmail(ValueError):
    pass


def parse(raw_email):
    # type: (Union[str, unicode]) -> Tuple[str, str]
    """Extract email from a full address. Example:
      'John Doe <jdoe+github@foo.com>' -> jdoe@foo.com

    >>> parse("John Doe <me+github.com@someorg.com")
    'me@someorg.com'
    >>> parse(42)
    Traceback (most recent call last):
        ...
    InvalidEmail
    """
    if not isinstance(raw_email, six.string_types):
        raise InvalidEmail
    if not raw_email or pd.isnull(raw_email):
        raise InvalidEmail
    email = raw_email.split("<", 1)[-1].split(">", 1)[0]
    chunks = email.split("@", 3)
    # git-svn generates emails with several @, e.g.:
    # <rossberg@chromium.org@ce2b1a6d-e550-0410-aec6-3dcde31c8c00>
    if len(chunks) < 2:
        raise InvalidEmail("Invalid email")
    uname = chunks[0].rsplit(" ", 1)[-1]
    addr_domain = chunks[1].split(" ", 1)[0]

    return uname.split("+", 1)[0], addr_domain


def clean(raw_email):
    # type: (Union[str, unicode]) -> Optional[str]
    """Extract email from a full address. Example:
      'John Doe <jdoe+github@foo.com>' -> jdoe@foo.com

    >>> clean(42) is None
    True
    >>> clean("John Doe me@someorg.com")
    'me@someorg.com'
    """
    try:
        return "%s@%s" % parse(raw_email)
    except InvalidEmail:
        return None


def domain(raw_email):
    # type: (Union[str, unicode]) -> Optional[str]
    """ Extract email domain from a raw email address.
    Returns None if the address is invalid

    >>> domain("John Doe <test@dep.uni.edu>")
    'dep.uni.edu'
    >>> domain("Missing test@dep.uni.edu@ce2b1a6d-e550-0410-aec6-3dcde31c8c00>")
    'dep.uni.edu'
    """
    try:
        return parse(raw_email)[-1]
    except InvalidEmail:
        return None


@memoize
def university_domains():
    # type: () -> set
    """ Return list of university domains outside of .edu TLD
    NOTE: only 2nd level domain is returned, i.e. for aaa.bbb.uk only bbbl.uk
          will be returned. This is necessary since many universities use
          departmenntal domains, like cs.cmu.edu or andrew.cmu.edu
    NOTE2: .edu domains are not included into this list as they're considered
          belonging to universities by default.

    How to get the original CSV:
    ```python
    x = requests.get(
         "https://raw.githubusercontent.com/Hipo/university-domains-list/"
         "master/world_universities_and_domains.json").json()
    domains = set(ds for u in x
                  for ds in u['domains'] if not "edu" in ds.rsplit(".", 2)[-2:])
    domains = list(domains)
    pd.Series(domains, index=domains, name="domain"
    ).drop(
        ["chat.ru"]
    ).to_csv("email_university_domains.csv", index=False)
    ```
    """
    fh = open(
        os.path.join(os.path.dirname(__file__), "email_university_domains.csv"))
    return set(addr_domain.strip() for addr_domain in fh)


@memoize
def public_domains():
    # type: () -> set
    """ Return list of public email domains (i.e. offering free mailboxes)

    How to get the original CSV:
    x = requests.get(
        "https://gist.githubusercontent.com/tbrianjones/5992856/raw/"
        "87f527af7bdd21997722fa65143a9af7bee92583/"
        "free_email_provider_domains.txt").text.split()
    # manually coded
    x.extend([
        'gmail.com', 'users.noreply.github.com', 'hotmail.com',
        'googlemail.com', 'users.sourceforge.net', 'iki.fi',
        'yahoo.com', 'me.com', 'gmx.de', 'cihar.com', 'ya.ru',
        'yandex.ru', 'outlook.com', 'gmx.net', 'web.de', 'pobox.com',
        'yahoo.co.uk', 'qq.com', 'free.fr', 'icloud.com', '163.com',
        '50mail.com', 'live.com', 'lavabit.com', 'mail.ru', '126.com',
        'yahoo.fr', 'seznam.cz'
    ])
    domains = list(set(x))  # make it unique
    pd.Series(domains, index=domains, name="domain"
    ).drop(  # mistakenly labeled as public
        ["unican.es"]
    ).to_csv("email_public_domains.csv", index=False)

    >>> 'gmail.com' in public_domains()
    True
    >>> '163.com' in public_domains()
    True
    >>> 'qq.com' in public_domains()
    True
    >>> 'jaraco.com' in public_domains()
    False
    """
    fh = open(
        os.path.join(os.path.dirname(__file__), "email_public_domains.csv"))
    return set(addr_domain.strip() for addr_domain in fh)


@memoize
def domain_user_stats():
    # type: () -> pd.Series
    """ Get number of distinct email addresses in observed domains

    TODO: get up to date with new projects layout
    How to build email_domain_users.csv:

    from collections import defaultdict
    import logging
    from common import utils as common
    import stscraper as scraper

    log = logging.getLogger("domain_user_stats")
    stats = defaultdict(set)
    for ecosystem in common.ECOSYSTEMS:
        urls = common.package_urls(ecosystem)
        for package_name, url in urls.items():
            log.info(package_name)
            try:
                cs = scraper.commits(url)
            except scraper.RepoDoesNotExist:
                continue
            for email_addr in cs["author_email"].dropna().unique():
                if not email_addr or pd.isnull(email_addr):
                    continue
                try:
                    user, email_domain = clean(email_addr).split("@")
                except InvalidEmail:
                    continue
                stats[email_domain].add(user)
    s = pd.Series({dm: len(users) for dm, users in stats.items()})
    s = s.rename("users").sort_values(ascending=False)
    s.to_csv("common/email_domain_users.csv", encoding="utf8", header=True)
    return s
    """
    fname = os.path.join(os.path.dirname(__file__), "email_domain_users.csv")
    stats = pd.read_csv(fname, header=0, squeeze=True, index_col=0)
    return stats[pd.notnull(stats.index)]


@memoize
def commercial_domains():
    # type: () -> set
    """ Return list of commercial email domains, which means:
    - domain is not public
    - domain is not university
    - it is not personal (more than 1 person using this domain)
    >>> "google.com" in commercial_domains()
    True
    >>> "microsoft.com" in commercial_domains()
    True
    >>> "isri.cs.cmu.edu" in commercial_domains()  # university department
    False
    >>> "jaraco.com" in commercial_domains()  # personal
    False
    """
    dus = domain_user_stats()
    es = "test@" + pd.Series(dus.index, index=dus.index)
    return set(
        dus[~is_public_bulk(es) & ~is_university_bulk(es) & (dus > 1)].index)


def is_university(addr):
    # type: (Union[str, unicode]) -> bool
    """ Check if provided email has a university domain

    - either in .edu domain
        (except public sercices like england.edu or australia.edu)
    - or in .edu.TLD (non-US based institutions, like edu.au)
    - or listed in a public list of universities
        since universities often have department addresses as well, only the end
        is matched. E.g. cs.cmu.edu will match cmu.edu

    :param addr: email address
    :return: bool
    >>> is_university("john@cmu.edu")
    True
    >>> is_university("john@gmail.com")
    False
    """
    addr_domain = domain(addr)
    if not addr_domain:  # invalid email
        return False
    chunks = addr_domain.split(".")
    if len(chunks) < 2:  # local or invalid address
        return False

    domains = university_domains()
    # many universitites have departmental emails, such as cs.cmu.edu. However,
    # the original dataset only has top level domain (cmu.edu). So, what we need
    # to do is to strip leading subdomains until match or nothing to strip:
    # isri.cs.cmu.edu (no match) -> cs.cmu.edu (no match) -> cmu.edu (match)
    return (chunks[-1] == "edu" and chunks[-2] not in ("england", "australia"))\
        or chunks[-2] == "edu" \
        or any(".".join(chunks[i:]) in domains for i in range(len(chunks)-1))


def is_public(addr):
    # type: (Union[str, unicode]) -> bool
    """ Check if the passed email registered at a free pubic mail server

    :param addr: email address to check
    :return: bool
    >>> is_public("john@cmu.edu")
    False
    >>> is_public("john@gmail.com")
    True
    """
    addr_domain = domain(addr)
    if not addr_domain:
        # anybody can use invalid email
        return True
    chunks = addr_domain.rsplit(".", 1)

    return len(chunks) < 2 \
        or addr_domain.endswith("local") \
        or addr_domain in public_domains()


def is_commercial(addr):
    """
    >>> is_commercial("test@google.com")
    True
    >>> is_commercial("test@microsoft.com")
    True
    >>> is_commercial("test@jaraco.com")
    False
    """
    addr_domain = domain(addr)
    return addr_domain and addr_domain in commercial_domains()


def is_commercial_bulk(addr_series):
    domains = commercial_domains()
    return addr_series.map(domain).map(lambda addr: addr in domains)


def is_public_bulk(addr_series):
    return addr_series.map(lambda addr: is_public(addr))


def is_university_bulk(addr_series):
    # type: (pd.Series) -> pd.Series
    """ Since university subdomains have to be matched by parts, we can't use
    bulk_check. Consider caching this call
    """
    return addr_series.map(lambda addr: is_university(addr))
