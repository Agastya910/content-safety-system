---
description: Clean up Docker resources and rebuild specific services
---

1. Stop all running containers
   ```bash
   docker-compose down
   ```

2. Prune stopped containers and dangling images (frees space)
   ```bash
   docker system prune -f
   ```
   *Note: Add `-a` if you want to remove ALL unused images, not just dangling ones.*

3. Rebuild and start the services
   ```bash
   docker-compose up -d --build
   ```
