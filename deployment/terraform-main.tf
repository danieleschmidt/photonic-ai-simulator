terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.primary_region
}

variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "regions" {
  description = "List of regions for multi-region deployment"
  type        = list(string)
  default     = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1", "eu-central-1", "ca-central-1"]
}

# ECS Cluster
resource "aws_ecs_cluster" "photonic_ai_cluster" {
  name = "photonic-ai-cluster"
  
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 1
  }
  
  tags = {
    Environment = "development"
    Application = "photonic-ai-system"
  }
}

# Application Load Balancer
resource "aws_lb" "photonic_ai_alb" {
  name               = "photonic-ai-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
  
  enable_deletion_protection = true
  
  tags = {
    Environment = "development"
    Application = "photonic-ai-system"
  }
}

# RDS Instance
resource "aws_db_instance" "photonic_ai_db" {
  identifier = "photonic-ai-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6g.xlarge"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "photonic_ai_prod"
  username = "photonic_user"
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  multi_az = true
  
  tags = {
    Environment = "development"
    Application = "photonic-ai-system"
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "photonic_ai_redis" {
  replication_group_id       = "photonic-ai-redis"
  description                = "Redis cluster for Photonic AI"
  
  node_type          = "cache.r6g.large"
  num_cache_clusters = 2
  
  port                     = 6379
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Environment = "development"
    Application = "photonic-ai-system"
  }
}

# Output important values
output "load_balancer_dns" {
  value = aws_lb.photonic_ai_alb.dns_name
}

output "database_endpoint" {
  value = aws_db_instance.photonic_ai_db.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.photonic_ai_redis.configuration_endpoint_address
}
