from pathlib import Path
import numpy as np
from PIL import Image

def print_tensor(x, name=''):
	try:
		print(f'{name:20} : {x.shape}, min : {x.min():.3f}, max : {x.max():.3f}, mean : {x.mean():.3f}, std : {x.std():.3f}')
	except:
		print(f'{name:20} : {x.shape}, min : {x.min():.3f}, max : {x.max():.3f}')


def find_objects(directory: Path, extensions=['.ply']) -> list:
	"""
	Recursively find all files with given extensions in a directory.
	"""
	asset_ids = set()
	for ext in extensions:
		for file_path in directory.rglob(f'*{ext}'):
			file_name = file_path.stem
			if '-' in file_name:
				asset_id = file_name.split('-')[0] # get part before first '-'
			elif '_' in file_name:
				asset_id = file_name.split('_')[0] # get part before first '_'
			else:
				asset_id = file_name
			asset_ids.add(asset_id)

	return list(asset_ids)


def get_render(asset_id, render_dir: Path = None) -> Image:
	"""
	Load the render image for a given asset ID.
	"""
	if render_dir is None:
		render_dir = Path('./data/gt_meshes/20251114_134358-gt_meshes-07-pipeline_65k/renders')
		if not render_dir.exists():
			raise FileNotFoundError(f'The default render directory not found: {render_dir}')
		
	render_path = render_dir / f'{asset_id}.png'
	if not render_path.exists():
		raise FileNotFoundError(f'Render image not found: {render_path}')
	image = Image.open(render_path)
	return image


def get_prim_mesh_path(asset_id, version: str, exp_name: str = None):
	"""
	Get the file path for a given asset ID and version (Gen or GT).

	args: 
		asset_id (str): The asset ID.
		version (str): 'Gen' or 'GT'.
		exp_name (str, optional): The experiment name for generated meshes. Defaults to None.
	"""
	# determine data directory
	if version.lower() == 'gt':
		data_dir = Path('./data/gt_meshes')
	elif version.lower() == 'gen':
		data_dir = Path('./data/gen_meshes')
	else:
		raise ValueError("version must be 'Gen' or 'GT'")
	if not data_dir.exists():
		raise FileNotFoundError(f'The data directory not found: {data_dir}. Please structure the data folder as follows: ./data/gt_meshes or ./data/gen_meshes depending on the version.')

	# determine experiment directory
	n_exps = len(list(data_dir.iterdir()))
	if n_exps > 1 and exp_name is None:
		raise ValueError(f'Multiple experiment directories found in {data_dir}. Please specify exp_name.')
	if exp_name is not None:
		exp_dir = data_dir / exp_name
	else:
		exp_dir = next(data_dir.iterdir())
	
	# get the folder that contains segmented meshes
	mesh_dir = exp_dir / 'segmented_meshes'
	if not mesh_dir.exists():
		raise FileNotFoundError(f'The segmented_meshes directory not found: {mesh_dir}. Please ensure the experiment directory contains a segmented_meshes folder.')
	
	mesh_path = mesh_dir / f'{asset_id}-prim.ply'
	if not mesh_path.exists():
		raise FileNotFoundError(f'Mesh file not found: {mesh_path}')
	
	return mesh_path