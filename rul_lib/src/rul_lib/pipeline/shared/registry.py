from typing import Any, Callable


def make_registry() -> tuple[dict[str, type], Callable[[str, type | None], Any]]:
  ''' Create a registry and a decorator to register classes.

      The decorator ensures that the class has a "params_cls" classmethod
      returning a class with a "from_dict" method.

      Returns
      -------
      registry:
        The registry mapping names to classes.
      register:
        The decorator to register classes.
  '''
  registry: dict[str, type] = {}

  def register(name: str, params_cls: type | None = None):

    def deco(cls):
      if name in registry:
        raise ValueError(f'registry key "{name}" already used by {registry[name].__name__}')
      declared = getattr(cls, 'name', None)
      if declared and declared not in (None, '', 'base') and declared != name:
        raise TypeError(f'{cls.__name__}: class "name"="{declared}" != registered "{name}"')
      cls.name = name
      setattr(cls, '_registry_key', name)
      P = params_cls or getattr(cls, 'ParamsCls', None)
      if P is None and hasattr(cls, 'params_cls') and callable(getattr(cls, 'params_cls')):
        P = cls.params_cls()
      if P is None or not hasattr(P, 'from_dict'):
        raise TypeError(f'{cls.__name__}: provide params_cls via decorator or ParamsCls or params_cls()')

      def _params_cls(_c):
        return P

      cls.params_cls = classmethod(_params_cls)
      registry[name] = cls
      return cls

    return deco

  return registry, register
