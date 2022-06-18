import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        reserved_ids = []

        for rect in objects_rect:  # Получаем точки прямоугольника детектированных объектов (далее (1))
            x, y, w, h = rect
            cx = (x + x + w) // 2  # Считаем центральные точки детектированного объекта
            cy = (y + y + h) // 2

            # Определяем: новый это объект или старый
            same_object_detected = False
            min_dist = 333
            min_id = 0
            for id, pt in self.center_points.items():  # для этого перебираем все объекты с предыдущей фотографии
                dist = math.hypot(cx - pt[0], cy - pt[1])  # высчитываем расстояние между (1) и старым объектом

                if dist < 80 and dist < min_dist and id not in reserved_ids:  # если расстояние меньше 40 пискелей, то это одинаковые объекты
                    min_dist = dist
                    min_id = id

            if min_dist < 80:
                reserved_ids.append(min_id)
                self.center_points[min_id] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, min_id])
                same_object_detected = True

            if same_object_detected is False:  # если до этого объект (1) ещё не встречался, то запомним его
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
