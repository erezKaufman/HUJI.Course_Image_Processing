import sol4

def main():
  office_panorama_generator = sol4.PanoramaGenerator('external/', 'office'  , 4)
  office_panorama_generator.generate_panorama()
  office_panorama_generator.show_panorama()

  backyard_panorama_generator = sol4.PanoramaGenerator('external/', 'backyard', 3)
  backyard_panorama_generator.generate_panorama()
  backyard_panorama_generator.show_panorama((20,10))

  oxford_panorama_generator = sol4.PanoramaGenerator('external/', 'oxford'  , 2)
  oxford_panorama_generator.generate_panorama()
  oxford_panorama_generator.show_panorama()


if __name__ == '__main__':
  main()
