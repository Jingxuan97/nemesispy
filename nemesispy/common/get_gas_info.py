from nemesispy.common.info_mol import mol_info
from nemesispy.common.info_mol_id import mol_id

def get_gas_name(id):
    """
    Find the name of the molecule given its ID number.

    Parameters
    ----------
    id : int
        ID of the molecule

    Returns
    -------
    name : str
        Name of the molecule.
    """
    id_str = str(id)
    try:
        name = mol_info[id_str]["name"]
    except KeyError:
        raise(Exception("Molecule ID {} not found.".format(id_str)))
    return name

def get_gas_id(name):
    """
    Find the ID of the molecule given its name.

    Parameters
    ----------
    name : str
        Name of the molecule

    Returns
    -------
    id : int
        ID of the molecule
    """
    try:
        id = mol_id[name]
    except KeyError:
        raise(Exception("Molecule name {} not found.".format(name)))
    return id
